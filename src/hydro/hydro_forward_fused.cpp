// kintera
#include <kintera/constants.h>

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
#include "primitive_projector_dispatch.hpp"

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

FusedPrimitiveProjector fused_projector(PrimitiveProjector const& pproj) {
  if (!pproj || pproj->options->type() == "none") {
    return FusedPrimitiveProjector::None;
  }
  if (pproj->options->type() == "density") {
    return FusedPrimitiveProjector::Density;
  }
  if (pproj->options->type() == "temperature") {
    return FusedPrimitiveProjector::Temperature;
  }
  TORCH_CHECK(false,
              "dynamics.fused-recon-riemann supports primitive projector "
              "types density and temperature, but got ",
              pproj->options->type());
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
    c10d::symmetric_memory::set_group_info(
        name, layout.options->process_rank(),
        layout.options->process_world_size(), layout.comm->store);
    initialized.insert(name);
  };

  // PyTorch's NVSHMEM allocator bootstraps anonymous allocations through this
  // default group; rendezvous below still uses the logical snap group.
  set_group_info_once("0");
  set_group_info_once(group_name);
}

void clear_fused_signal_slots(
    c10::intrusive_ptr<c10d::symmetric_memory::SymmetricMemory> const& symm,
    std::shared_ptr<ProcessGroupContext> const& comm) {
  auto signal_pad = symm->get_signal_pad(
      symm->get_rank(), {2 * symm->get_world_size()}, torch::kInt32);
  signal_pad.zero_();
  (void)signal_pad.sum().item<int>();
  if (comm) comm->barrier();
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
    pool.self_ptr = torch::tensor({base}, torch::dtype(torch::kInt64)).to(device);
  }
  auto [inserted, _] = pools.emplace(key, std::move(pool));
  return inserted->second;
}

//! \brief Reusable process-local barrier over the blocks_per_process worker
//! threads. The fused cubed-sphere exchange runs pack -> sync -> flux -> release
//! on one shared symmetric buffer; these barriers guarantee every local block
//! has enqueued its pack before the single cross-process visibility sync, and
//! that the sync is enqueued before any block reads peer state. All exchange
//! kernels share the device's default stream, so enqueue order fixed by these
//! barriers is the execution order, which removes the single-block launch
//! ordering deadlock that made the old path require blocks_per_process == 1.
class FusedExchangeBarrier {
 public:
  explicit FusedExchangeBarrier(int participants) : participants_(participants) {}
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

torch::Tensor make_side_meta(CubedSphereLayoutImpl const& layout,
                             bool exchange_dim2, bool exchange_dim3,
                             torch::Device device) {
  // Per side: [enabled, peer_process, peer_local_block, peer_side, rev].
  // neighbor_rank returns a global BLOCK rank; translate it into the owning
  // process (to index the symmetric buffer_ptrs array) and the slot within that
  // process (to index the shared buffer), so the exchange addresses same-GPU and
  // cross-GPU peers uniformly. Must match the CS_META_* layout in the kernels.
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
    TORCH_CHECK(playout->options->process_world_size() == 1 ||
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
  auto projector = fused_projector(pproj);

  torch::Tensor inv_mu_ratio_m1, cv_ratio_m1, u0;
  if (eos == FusedEos::IdealMoist) {
    auto ideal_moist = dynamic_cast<IdealMoistImpl const*>(peos.get());
    TORCH_CHECK(ideal_moist != nullptr,
                "dynamics.fused-recon-riemann expected IdealMoistImpl");
    inv_mu_ratio_m1 = ideal_moist->inv_mu_ratio_m1.to(w.options());
    cv_ratio_m1 = ideal_moist->cv_ratio_m1.to(w.options());
    u0 = ideal_moist->u0.to(w.options());
  }

  torch::Tensor rho_grav = torch::zeros_like(w[IDN]);
  torch::Tensor w1 = w;
  torch::Tensor psf, dx1f;
  double gas_constant = 0.;
  if (projector != FusedPrimitiveProjector::None) {
    TORCH_CHECK(eos != FusedEos::ShallowWater,
                "dynamics.fused-recon-riemann primitive projectors are not "
                "defined for shallow-water EOS");
    TORCH_CHECK(options->grav(),
                "dynamics.fused-recon-riemann primitive projector requires "
                "const-gravity forcing");
    w1 = torch::empty_like(w);
    psf = torch::empty({w.size(1), w.size(2), w.size(3) + 1}, w.options());
    dx1f = pmb->pcoord->dx1f.to(w.options());
    if (projector == FusedPrimitiveProjector::Temperature) {
      gas_constant = kintera::constants::Rgas / peos->species_weight();
    }
    primitive_projector_dispatch(
        w, w1, psf, dx1f, pmb->pcoord->il(), pmb->pcoord->iu() + 1, projector,
        -options->grav()->grav1(), pproj->options->margin(), gas_constant);
  }

  if (u.size(3) > 1 && !options->disable_flux_x1()) {
    fused_recon_riemann_cuda(
        w1, _flux1, /*dim=*/3, recon1_prim, recon1_vel, solver, eos,
        options->eos()->gammad(), options->eos()->density_floor(),
        options->eos()->pressure_floor(), options->eos()->limiter(),
        inv_mu_ratio_m1, cv_ratio_m1, u0, shallow_roe_dir_yz, projector, psf,
        dx1f, gas_constant, rho_grav, cubed_sphere_layout, face, x2v, x2f,
        x3v, x3f);
  }
  if (u.size(3) > 1 && psed) {
    psed->forward(w, _flux1);
  }
  if (u.size(2) > 1 && !options->disable_flux_x2()) {
    fused_recon_riemann_cuda(
        w, _flux2, /*dim=*/2, recon23_prim, recon23_vel, solver, eos,
        options->eos()->gammad(), options->eos()->density_floor(),
        options->eos()->pressure_floor(), options->eos()->limiter(),
        inv_mu_ratio_m1, cv_ratio_m1, u0, shallow_roe_dir_yz,
        FusedPrimitiveProjector::None, torch::Tensor(), torch::Tensor(), 0.,
        torch::Tensor(), cubed_sphere_layout, face, x2v, x2f, x3v, x3f);
  }
  if (u.size(1) > 1 && !options->disable_flux_x3()) {
    fused_recon_riemann_cuda(
        w, _flux3, /*dim=*/1, recon23_prim, recon23_vel, solver, eos,
        options->eos()->gammad(), options->eos()->density_floor(),
        options->eos()->pressure_floor(), options->eos()->limiter(),
        inv_mu_ratio_m1, cv_ratio_m1, u0, shallow_roe_dir_yz,
        FusedPrimitiveProjector::None, torch::Tensor(), torch::Tensor(), 0.,
        torch::Tensor(), cubed_sphere_layout, face, x2v, x2f, x3v, x3f);
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
          nvar * edge_len * nc1, edge_len * nc1, nc1, 1};

      if (multi_process) ensure_symmetric_group(*playout, group_name);
      auto& pool = get_fused_symmetric_exchange_pool(
          group_name, w.scalar_type(), w.device(), sizes, strides,
          multi_process);
      auto side_meta =
          make_side_meta(*cs_layout, exchange_dim2, exchange_dim3, w.device());
      auto& barrier = get_fused_exchange_barrier(group_name, bpp);

      // (1) every local block reconstructs its panel-edge states into its slice
      fused_cubed_sphere_pack_cuda(
          w, pool.buffer, side_meta, face, local_block, x2v, x2f, x3v, x3f,
          recon23_prim, recon23_vel, eos, options->eos()->density_floor(),
          options->eos()->pressure_floor(), options->eos()->limiter());
      barrier.wait();

      // (2) one block per process publishes cross-process write visibility
      if (multi_process && is_leader) {
        clear_fused_signal_slots(pool.symm, playout->comm);
        fused_cubed_sphere_sync_cuda(pool.signal_pads_dev(), pool.rank(),
                                     pool.world_size(), w.device());
      }
      barrier.wait();

      // (3) every local block overwrites its cross-panel boundary flux from own
      //     + peer edge states (peer buffer selected by owning process rank)
      fused_cubed_sphere_flux_cuda(
          w, _flux2, _flux3, pool.buffer, pool.buffer_ptrs_dev(), side_meta,
          face, local_block, x2v, x2f, x3v, x3f, recon23_prim, recon23_vel,
          solver, eos, options->eos()->gammad(),
          options->eos()->density_floor(), options->eos()->pressure_floor(),
          options->eos()->limiter(), inv_mu_ratio_m1, cv_ratio_m1, u0,
          shallow_roe_dir_yz);
      barrier.wait();

      // (4) one block per process closes the read epoch before the next step
      if (multi_process && is_leader) {
        fused_cubed_sphere_release_cuda(pool.signal_pads_dev(), pool.rank(),
                                        pool.world_size(), w.device());
      }
      barrier.wait();
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
  return du;
}

}  // namespace snap
