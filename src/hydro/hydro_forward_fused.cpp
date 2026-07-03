// C/C++
#include <algorithm>
#include <array>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <set>
#include <string>
#include <unordered_map>

// snap
#include <snap/snap.h>

#include <snap/layout/cubed_sphere_layout.hpp>
#include <snap/layout/layout.hpp>
#include <snap/mesh/meshblock.hpp>
#include <torch/csrc/distributed/c10d/symm_mem/SymmetricMemory.hpp>

#include "../eos/ideal_moist.hpp"
#include "../sedimentation/sedimentation.hpp"
#include "fused_recon_riemann_dispatch.hpp"
#include "hydro.hpp"

namespace snap {
namespace {

FusedReconScheme fused_recon_scheme(std::string const& type,
                                    bool velocity_group) {
  if (type == "weno5") {
    return velocity_group ? FusedReconScheme::CP5 : FusedReconScheme::WENO5;
  }
  if (type == "weno3") {
    return velocity_group ? FusedReconScheme::CP3 : FusedReconScheme::WENO3;
  }
  if (type == "cp5") return FusedReconScheme::CP5;
  if (type == "cp3") return FusedReconScheme::CP3;
  TORCH_CHECK(false,
              "dynamics.fused-recon-riemann supports cp3, cp5, weno3, and "
              "weno5 reconstruction, but got ",
              type);
}

bool fused_combo_supported(std::string const& eos_type,
                           std::string const& riemann_type) {
  return ((eos_type == "ideal-gas" || eos_type == "ideal-moist") &&
          (riemann_type == "lmars" || riemann_type == "hllc")) ||
         (eos_type == "shallow-water" && riemann_type == "shallow-roe");
}

FusedEos fused_eos(std::string const& type) {
  if (type == "ideal-gas") return FusedEos::IdealGas;
  if (type == "ideal-moist") return FusedEos::IdealMoist;
  if (type == "shallow-water") return FusedEos::ShallowWater;
  TORCH_CHECK(false,
              "dynamics.fused-recon-riemann supports EOS types ideal-gas, "
              "ideal-moist, and shallow-water, but got ",
              type);
}

FusedRiemannSolver fused_riemann_solver(std::string const& type) {
  if (type == "lmars") return FusedRiemannSolver::LMARS;
  if (type == "hllc") return FusedRiemannSolver::HLLC;
  if (type == "shallow-roe") return FusedRiemannSolver::ShallowRoe;
  TORCH_CHECK(false,
              "dynamics.fused-recon-riemann supports Riemann solvers lmars, "
              "hllc, and shallow-roe, but got ",
              type);
}

void ensure_symmetric_group(LayoutImpl const& layout,
                            std::string const& group_name) {
  static std::mutex mutex;
  static std::set<std::string> initialized;
  std::lock_guard<std::mutex> lock(mutex);
  if (initialized.count(group_name)) return;
  TORCH_CHECK(layout.comm != nullptr && layout.comm->store.defined(),
              "dynamics.fused-recon-riemann cubed-sphere exchange requires an "
              "initialized process-group store");
  if (initialized.empty()) {
    c10d::symmetric_memory::set_backend("NVSHMEM");
    c10d::symmetric_memory::set_signal_pad_size(std::max<size_t>(
        1024, 4 * layout.options->process_world_size() * sizeof(uint32_t)));
  }

  auto set_group_info_once = [&](std::string const& name) {
    if (initialized.count(name)) return;
    c10d::symmetric_memory::set_group_info(name, layout.options->process_rank(),
                                           layout.options->process_world_size(),
                                           layout.comm->store);
    initialized.insert(name);
  };

  // PyTorch's NVSHMEM allocator bootstraps anonymous allocations through this
  // default group; rendezvous below still uses the logical snap group.
  set_group_info_once("0");
  set_group_info_once(group_name);
}

struct FusedSymmetricExchangePool {
  torch::Tensor buffer;
  //! non-null when the exchange spans multiple processes (NVSHMEM P2P)
  c10::intrusive_ptr<c10d::symmetric_memory::SymmetricMemory> symm;
  //! single-process fallback: a device int64 tensor holding buffer.data_ptr()
  //! so the flux kernel can index buf_ptrs[0] uniformly for same-GPU peers.
  torch::Tensor self_ptr;

  void** buffer_ptrs_dev() const {
    if (symm) return symm->get_buffer_ptrs_dev();
    return reinterpret_cast<void**>(self_ptr.data_ptr<int64_t>());
  }
  uint32_t** signal_pads_dev() const {
    return symm ? reinterpret_cast<uint32_t**>(symm->get_signal_pad_ptrs_dev())
                : nullptr;
  }
  int rank() const { return symm ? symm->get_rank() : 0; }
  int world_size() const { return symm ? symm->get_world_size() : 1; }
};

uint64_t stable_alloc_id(std::string const& key) {
  uint64_t hash = 1469598103934665603ull;
  for (unsigned char c : key) {
    hash ^= c;
    hash *= 1099511628211ull;
  }
  return hash == 0 ? 1 : hash;
}

FusedSymmetricExchangePool& get_fused_symmetric_exchange_pool(
    std::string const& group_name, c10::ScalarType dtype, torch::Device device,
    std::vector<int64_t> const& sizes, std::vector<int64_t> const& strides,
    bool use_symmetric) {
  static std::mutex mutex;
  static std::unordered_map<std::string, FusedSymmetricExchangePool> pools;

  std::string key = group_name + ":device=" + std::to_string(device.index()) +
                    ":dtype=" + std::to_string(static_cast<int>(dtype));
  for (auto size : sizes) key += ":s" + std::to_string(size);
  for (auto stride : strides) key += ":t" + std::to_string(stride);

  std::lock_guard<std::mutex> lock(mutex);
  auto it = pools.find(key);
  if (it != pools.end()) return it->second;

  FusedSymmetricExchangePool pool;
  if (use_symmetric) {
    pool.buffer = c10d::symmetric_memory::empty_strided_p2p(
        sizes, strides, dtype, device, std::nullopt, stable_alloc_id(key));
    pool.symm = c10d::symmetric_memory::rendezvous(pool.buffer, group_name);
  } else {
    // Single process: all panels are co-resident on one GPU, so a plain
    // contiguous buffer read by every local block (buf_ptrs[0]) is sufficient;
    // no NVSHMEM group or signal-pad handshake is required.
    pool.buffer =
        torch::empty(sizes, torch::TensorOptions().dtype(dtype).device(device));
    int64_t base = reinterpret_cast<int64_t>(pool.buffer.data_ptr());
    pool.self_ptr =
        torch::tensor({base}, torch::dtype(torch::kInt64)).to(device);
  }
  auto [inserted, _] = pools.emplace(key, std::move(pool));
  return inserted->second;
}

//! \brief Reusable process-local barrier over the blocks_per_process worker
//! threads. The fused cubed-sphere exchange runs pack -> sync -> flux ->
//! release on one shared symmetric buffer; these barriers guarantee every local
//! block has enqueued its pack before the single cross-process visibility sync,
//! and that the sync is enqueued before any block reads peer state. All
//! exchange kernels share the device's default stream, so enqueue order fixed
//! by these barriers is the execution order, which removes the single-block
//! launch ordering deadlock that made the old path require blocks_per_process
//! == 1.
class FusedExchangeBarrier {
 public:
  explicit FusedExchangeBarrier(int participants)
      : participants_(participants) {}
  void wait() {
    std::unique_lock<std::mutex> lock(mutex_);
    int gen = generation_;
    if (++arrived_ == participants_) {
      arrived_ = 0;
      ++generation_;
      cv_.notify_all();
    } else {
      cv_.wait(lock, [&] { return generation_ != gen; });
    }
  }

 private:
  std::mutex mutex_;
  std::condition_variable cv_;
  int participants_;
  int arrived_ = 0;
  int generation_ = 0;
};

FusedExchangeBarrier& get_fused_exchange_barrier(std::string const& key,
                                                 int participants) {
  static std::mutex mutex;
  static std::unordered_map<std::string, std::unique_ptr<FusedExchangeBarrier>>
      barriers;
  std::lock_guard<std::mutex> lock(mutex);
  auto it = barriers.find(key);
  if (it == barriers.end()) {
    it = barriers
             .emplace(key, std::make_unique<FusedExchangeBarrier>(participants))
             .first;
  }
  return *it->second;
}

//! \brief Per-process collection of the local panels' flux tensors / face ids /
//! side metadata, so one leader-launched kernel can overwrite every panel's
//! cross-panel boundary flux. Each panel writes its own local-block slot before
//! the barrier; the leader assembles the device pointer arrays after it. The
//! device arrays are kept alive here (persistent) so the async flux-all kernel
//! never reads a freed pointer buffer.
struct FusedSeamBatch {
  //! _flux2/_flux3 are persistent HydroImpl buffers, so raw ptrs are stable.
  std::vector<int64_t> flux2_ptr, flux3_ptr;  // per local block
  std::vector<int32_t> faces;                 // per local block
  //! side_meta / coords are transient per call; hold the tensors so their
  //! device pointers stay valid until the async flux-all kernel completes.
  std::vector<torch::Tensor> side_meta, x2v, x2f, x3v, x3f;
  torch::Tensor flux2_dev, flux3_dev, faces_dev, side_meta_all;
  torch::Tensor x2v_dev, x2f_dev, x3v_dev, x3f_dev;
};

FusedSeamBatch& get_fused_seam_batch(std::string const& key, int bpp) {
  static std::mutex mutex;
  static std::unordered_map<std::string, FusedSeamBatch> batches;
  std::lock_guard<std::mutex> lock(mutex);
  auto& b = batches[key];
  if (static_cast<int>(b.faces.size()) != bpp) {
    b.flux2_ptr.assign(bpp, 0);
    b.flux3_ptr.assign(bpp, 0);
    b.faces.assign(bpp, 0);
    b.side_meta.assign(bpp, torch::Tensor());
    b.x2v.assign(bpp, torch::Tensor());
    b.x2f.assign(bpp, torch::Tensor());
    b.x3v.assign(bpp, torch::Tensor());
    b.x3f.assign(bpp, torch::Tensor());
  }
  return b;
}

//! One-time signal-pad zero after rendezvous. sync_remote_blocks self-restores
//! the pad to zero after each sync+release, so the pad only needs clearing once
//! (removing the per-substage .item() host<->device sync).
void clear_fused_signal_slots_once(
    c10::intrusive_ptr<c10d::symmetric_memory::SymmetricMemory> const& symm,
    std::shared_ptr<ProcessGroupContext> const& comm,
    std::string const& group_name) {
  static std::mutex mutex;
  static std::set<std::string> cleared;
  {
    std::lock_guard<std::mutex> lock(mutex);
    if (cleared.count(group_name)) return;
    cleared.insert(group_name);
  }
  auto signal_pad = symm->get_signal_pad(
      symm->get_rank(), {2 * symm->get_world_size()}, torch::kInt32);
  signal_pad.zero_();
  (void)signal_pad.sum().item<int>();  // one-time only
  if (comm) comm->barrier();
}

torch::Tensor make_side_meta(CubedSphereLayoutImpl const& layout,
                             bool exchange_dim2, bool exchange_dim3,
                             torch::Device device) {
  // Per side: [enabled, peer_process, peer_local_block, peer_side, rev].
  // neighbor_rank returns a global BLOCK rank; translate it into the owning
  // process (to index the symmetric buffer_ptrs array) and the slot within that
  // process (to index the shared buffer), so the exchange addresses same-GPU
  // and cross-GPU peers uniformly. Must match the CS_META_* layout in the
  // kernels.
  constexpr int kStride = 5;
  std::vector<int> meta(4 * kStride, 0);
  auto iloc = layout.loc_of(layout.options->rank());
  int face = std::get<2>(iloc);
  std::array<std::tuple<int, int, int>, 4> offsets = {
      std::tuple<int, int, int>{0, -1, 0},
      std::tuple<int, int, int>{0, +1, 0},
      std::tuple<int, int, int>{-1, 0, 0},
      std::tuple<int, int, int>{+1, 0, 0},
  };
  for (int side = 0; side < 4; ++side) {
    bool dim_enabled = side <= SIDE_R ? exchange_dim2 : exchange_dim3;
    if (!dim_enabled) continue;
    int nb = layout.neighbor_rank(iloc, offsets[side]);
    if (nb < 0) continue;
    auto nb_loc = layout.loc_of(nb);
    if (std::get<2>(nb_loc) == face) continue;
    auto edge = CS_FACE_EDGES[face][side];
    meta[side * kStride + 0] = 1;
    meta[side * kStride + 1] = layout.options->owner_process_rank(nb);
    meta[side * kStride + 2] = layout.options->local_block_index(nb);
    meta[side * kStride + 3] = edge.nside;
    meta[side * kStride + 4] = edge.rev;
  }
  return torch::tensor(meta, torch::dtype(torch::kInt32)).to(device);
}

}  // namespace

torch::Tensor HydroImpl::_forward_fused(double dt, torch::Tensor u,
                                        Variables const& other) {
#ifdef NOT_USE_CUDA
  TORCH_CHECK(false,
              "dynamics.fused-recon-riemann requires a CUDA-enabled build");
#endif

  TORCH_CHECK(u.is_cuda(),
              "dynamics.fused-recon-riemann requires CUDA tensors");
  TORCH_CHECK(other.count("hydro_w"),
              "dynamics.fused-recon-riemann requires hydro_w primitives");
  auto const& w = other.at("hydro_w");
  TORCH_CHECK(w.is_cuda(),
              "dynamics.fused-recon-riemann requires CUDA primitive tensors");
  TORCH_CHECK(w.is_contiguous() && u.is_contiguous(),
              "dynamics.fused-recon-riemann requires contiguous hydro tensors");
  TORCH_CHECK(_flux1.is_contiguous() &&
                  (!_flux2.defined() || _flux2.is_contiguous()) &&
                  (!_flux3.defined() || _flux3.is_contiguous()),
              "dynamics.fused-recon-riemann requires contiguous flux buffers");
  TORCH_CHECK(w.size(0) <= 64,
              "dynamics.fused-recon-riemann supports at most 64 hydro "
              "variables, but got ",
              w.size(0));
  TORCH_CHECK(w.size(0) - ICY <= 32,
              "dynamics.fused-recon-riemann supports at most 32 mass "
              "fractions, but got ",
              w.size(0) - ICY);
  TORCH_CHECK(!other.count("solid"),
              "dynamics.fused-recon-riemann does not yet support solid "
              "internal-boundary state revision");

  auto eos_type = options->eos()->type();
  auto riemann_type = options->riemann()->type();
  TORCH_CHECK(fused_combo_supported(eos_type, riemann_type),
              "dynamics.fused-recon-riemann does not support EOS/Riemann "
              "combination ",
              eos_type, "/", riemann_type);

  auto playout = pmb->get_layout();
  bool cubed_sphere_layout = playout->options->type() == "cubed-sphere";
  if (!cubed_sphere_layout) {
    TORCH_CHECK(pmb->pcoord->options->type() == "cartesian",
                "dynamics.fused-recon-riemann currently supports cartesian "
                "coordinates or cubed-sphere layouts only, but got coordinate "
                "type ",
                pmb->pcoord->options->type(), " with layout type ",
                playout->options->type());
  } else {
    // Multi-block-per-process is supported: each process may own several panels
    // co-resident on one GPU. A single shared per-process symmetric buffer is
    // sliced by local block, and cross-panel peers are addressed by (owning
    // process, local block). Cross-process exchange still needs the NVSHMEM
    // process group; a single-process run (all panels on one GPU) does not.
    TORCH_CHECK(
        playout->options->process_world_size() == 1 ||
            playout->has_process_group(),
        "dynamics.fused-recon-riemann cubed-sphere exchange requires an "
        "initialized process group for multi-process runs");
  }

  CubedSphereLayoutImpl const* cs_layout = nullptr;
  int face = 0;
  torch::Tensor x2v, x2f, x3v, x3f;
  if (cubed_sphere_layout) {
    cs_layout = dynamic_cast<CubedSphereLayoutImpl const*>(playout.get());
    TORCH_CHECK(cs_layout != nullptr,
                "expected CubedSphereLayoutImpl for cubed-sphere layout");
    face = std::get<2>(cs_layout->loc_of(cs_layout->options->rank()));
    x2v = pmb->pcoord->x2v.to(w.options());
    x2f = pmb->pcoord->x2f.to(w.options());
    x3v = pmb->pcoord->x3v.to(w.options());
    x3f = pmb->pcoord->x3f.to(w.options());
  }

  peos->forward(u, w);

  auto recon1_prim = fused_recon_scheme(precon1->pinterp1->options->type(),
                                        /*velocity_group=*/false);
  auto recon1_vel = fused_recon_scheme(precon1->pinterp1->options->type(),
                                       /*velocity_group=*/true);
  auto recon23_prim = fused_recon_scheme(precon23->pinterp1->options->type(),
                                         /*velocity_group=*/false);
  auto recon23_vel = fused_recon_scheme(precon23->pinterp1->options->type(),
                                        /*velocity_group=*/true);
  auto solver = fused_riemann_solver(riemann_type);
  auto eos = fused_eos(eos_type);
  int shallow_roe_dir_yz = options->riemann()->dir() == "yz" ? 1 : 0;

  torch::Tensor inv_mu_ratio_m1, cv_ratio_m1, u0;
  int nvapor = 0;
  if (eos == FusedEos::IdealMoist) {
    auto ideal_moist = dynamic_cast<IdealMoistImpl const*>(peos.get());
    TORCH_CHECK(ideal_moist != nullptr,
                "dynamics.fused-recon-riemann expected IdealMoistImpl");
    inv_mu_ratio_m1 = ideal_moist->inv_mu_ratio_m1.to(w.options());
    cv_ratio_m1 = ideal_moist->cv_ratio_m1.to(w.options());
    u0 = ideal_moist->u0.to(w.options());
    nvapor =
        static_cast<int>(ideal_moist->pthermo->options->vapor_ids().size()) - 1;
  }

  torch::Tensor rho_grav = torch::zeros_like(w[IDN]);
  bool revise_x1_lr = options->grav() && (options->grav()->grav1() != 0);
  torch::Tensor dx1f;

  if (u.size(3) > 1) {
    if (revise_x1_lr) {
      _revise_x1inner_ghost(w);
      _revise_x1outer_ghost(w);
      dx1f = pmb->pcoord->dx1f.to(w.options());
    }
  }

  auto make_physics_params = [&](FusedReconScheme recon_prim,
                                 FusedReconScheme recon_vel) {
    return FusedPhysicsParams{recon_prim,
                              recon_vel,
                              solver,
                              eos,
                              options->eos()->gammad(),
                              options->eos()->density_floor(),
                              options->eos()->pressure_floor(),
                              options->eos()->limiter(),
                              inv_mu_ratio_m1,
                              cv_ratio_m1,
                              u0,
                              nvapor,
                              shallow_roe_dir_yz};
  };
  FusedMetricParams metric_params{
      cubed_sphere_layout, face, x2v, x2f, x3v, x3f};

  if (u.size(3) > 1 && !options->disable_flux_x1()) {
    fused_recon_riemann_cuda(
        w, _flux1,
        FusedReconRiemannParams{
            /*dim=*/3, make_physics_params(recon1_prim, recon1_vel),
            FusedX1RevisionParams{revise_x1_lr, dx1f, rho_grav},
            metric_params});
  }
  if (u.size(3) > 1 && psed) {
    psed->forward(w, _flux1);
  }
  if (u.size(2) > 1 && !options->disable_flux_x2()) {
    fused_recon_riemann_cuda(
        w, _flux2,
        FusedReconRiemannParams{
            /*dim=*/2, make_physics_params(recon23_prim, recon23_vel),
            FusedX1RevisionParams{false, torch::Tensor(), torch::Tensor()},
            metric_params});
  }
  if (u.size(1) > 1 && !options->disable_flux_x3()) {
    fused_recon_riemann_cuda(
        w, _flux3,
        FusedReconRiemannParams{
            /*dim=*/1, make_physics_params(recon23_prim, recon23_vel),
            FusedX1RevisionParams{false, torch::Tensor(), torch::Tensor()},
            metric_params});
  }

  if (cubed_sphere_layout) {
    bool exchange_dim2 = u.size(2) > 1 && !options->disable_flux_x2();
    bool exchange_dim3 = u.size(1) > 1 && !options->disable_flux_x3();
    if (exchange_dim2 || exchange_dim3) {
      std::string group_name = "snapy:fused-recon-riemann:cubed-sphere";
      int bpp = std::max(1, playout->options->blocks_per_process());
      int local_block =
          playout->options->local_block_index(playout->options->rank());
      bool multi_process = playout->options->process_world_size() > 1;
      bool is_leader = local_block == 0;

      int edge_len = std::max<int>(w.size(1), w.size(2));
      constexpr int kSides = 4;
      constexpr int kStates = 2;
      int64_t nvar = w.size(0), nc1 = w.size(3);
      // shared per-process buffer: [blocks_per_process, side, state, var, edge,
      // nc1]; every local block owns slice `local_block`.
      std::vector<int64_t> sizes = {bpp, kSides, kStates, nvar, edge_len, nc1};
      std::vector<int64_t> strides = {
          static_cast<int64_t>(kSides) * kStates * nvar * edge_len * nc1,
          static_cast<int64_t>(kStates) * nvar * edge_len * nc1,
          nvar * edge_len * nc1,
          edge_len * nc1,
          nc1,
          1};

      if (multi_process) ensure_symmetric_group(*playout, group_name);
      auto& pool = get_fused_symmetric_exchange_pool(
          group_name, w.scalar_type(), w.device(), sizes, strides,
          multi_process);
      auto side_meta =
          make_side_meta(*cs_layout, exchange_dim2, exchange_dim3, w.device());
      auto& barrier = get_fused_exchange_barrier(group_name, bpp);
      auto& batch = get_fused_seam_batch(group_name, bpp);

      // register this panel's flux tensors / face / coords / side_meta for the
      // single process-level seam-flux launch (distinct local-block slots, so
      // the concurrent workers never touch the same entry)
      batch.flux2_ptr[local_block] =
          reinterpret_cast<int64_t>(_flux2.data_ptr());
      batch.flux3_ptr[local_block] =
          reinterpret_cast<int64_t>(_flux3.data_ptr());
      batch.faces[local_block] = face;
      batch.side_meta[local_block] = side_meta;
      batch.x2v[local_block] = x2v;
      batch.x2f[local_block] = x2f;
      batch.x3v[local_block] = x3v;
      batch.x3f[local_block] = x3f;

      // ---- Phase A: every local panel packs its edge states into its slice
      // ----
      FusedCubedSpherePanelParams panel_params{
          side_meta, face, local_block, x2v, x2f, x3v, x3f};
      fused_cubed_sphere_pack_cuda(
          w, pool.buffer,
          FusedCubedSpherePackParams{panel_params, recon23_prim, recon23_vel,
                                     eos, options->eos()->density_floor(),
                                     options->eos()->pressure_floor(),
                                     options->eos()->limiter()});
      barrier.wait();  // all panels packed + registered

      // ---- Phase B: one leader per process runs the whole seam exchange on
      // the GPU: cross-process visibility sync -> ONE flux kernel over ALL
      // local panels (overwrites each panel's cross-panel boundary flux) ->
      // close the read epoch. No per-panel flux launches and no per-substage
      // .item(). ----
      if (is_leader) {
        if (multi_process) {
          clear_fused_signal_slots_once(pool.symm, playout->comm, group_name);
          fused_cubed_sphere_sync_cuda(pool.signal_pads_dev(), pool.rank(),
                                       pool.world_size(), w.device());
        }
        auto i64 = torch::dtype(torch::kInt64);
        auto ptr_tensor = [&](std::vector<torch::Tensor> const& ts) {
          std::vector<int64_t> ptrs(ts.size());
          for (size_t p = 0; p < ts.size(); ++p)
            ptrs[p] = reinterpret_cast<int64_t>(ts[p].data_ptr());
          return torch::tensor(ptrs, i64).to(w.device());
        };
        batch.flux2_dev = torch::tensor(batch.flux2_ptr, i64).to(w.device());
        batch.flux3_dev = torch::tensor(batch.flux3_ptr, i64).to(w.device());
        batch.x2v_dev = ptr_tensor(batch.x2v);
        batch.x2f_dev = ptr_tensor(batch.x2f);
        batch.x3v_dev = ptr_tensor(batch.x3v);
        batch.x3f_dev = ptr_tensor(batch.x3f);
        batch.faces_dev =
            torch::tensor(batch.faces, torch::dtype(torch::kInt32))
                .to(w.device());
        batch.side_meta_all = torch::stack(batch.side_meta, 0);
        auto as_pp = [](torch::Tensor& t) {
          return reinterpret_cast<void**>(t.data_ptr<int64_t>());
        };
        fused_cubed_sphere_flux_all_cuda(
            pool.buffer,
            FusedCubedSphereFluxAllPtrs{
                as_pp(batch.flux2_dev), as_pp(batch.flux3_dev),
                pool.buffer_ptrs_dev(), as_pp(batch.x2v_dev),
                as_pp(batch.x2f_dev), as_pp(batch.x3v_dev),
                as_pp(batch.x3f_dev)},
            FusedCubedSphereFluxAllParams{
                batch.side_meta_all, batch.faces_dev, static_cast<int>(nvar),
                static_cast<int>(w.size(1)), static_cast<int>(w.size(2)),
                static_cast<int>(nc1), bpp, w.scalar_type(), w.device(),
                make_physics_params(recon23_prim, recon23_vel)});
        if (multi_process) {
          fused_cubed_sphere_release_cuda(pool.signal_pads_dev(), pool.rank(),
                                          pool.world_size(), w.device());
        }
      }
      barrier.wait();  // seam flux enqueued before any panel's divergence
    }
  }
  _div.set_(pmb->pcoord->forward(w, _flux1, _flux2, _flux3));

  auto du = torch::zeros_like(_div);
  auto interior = pmb->part({0, 0, 0}, PartOptions().exterior(false));
  du.index(interior) = -dt * _div.index(interior);

  auto temp = peos->compute("W->T", {w});
  for (auto& f : forcings) f.forward(du, w, temp, dt);

  if (options->grav() && (options->grav()->non_hydrostatic() < 1.)) {
    du[IVX] += dt * rho_grav * (1. - options->grav()->non_hydrostatic());
    du[IPR] +=
        dt * w[IVX] * rho_grav * (1. - options->grav()->non_hydrostatic());
  }
  _apply_implicit_correction(du, w, dt, other);
  return du;
}

}  // namespace snap
