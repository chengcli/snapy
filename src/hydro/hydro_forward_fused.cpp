// kintera
#include <kintera/constants.h>

// C/C++
#include <algorithm>
#include <array>
#include <mutex>
#include <set>

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

torch::Tensor make_side_meta(CubedSphereLayoutImpl const& layout,
                             bool exchange_dim2, bool exchange_dim3,
                             torch::Device device) {
  constexpr int kStride = 4;
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
    meta[side * kStride + 1] = nb;
    meta[side * kStride + 2] = edge.nside;
    meta[side * kStride + 3] = edge.rev;
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
    TORCH_CHECK(
        playout->options->blocks_per_process() == 1,
        "dynamics.fused-recon-riemann cubed-sphere symmetric-memory exchange "
        "currently requires blocks_per_process=1 to avoid local-block launch "
        "ordering deadlock, but got ",
        playout->options->blocks_per_process());
    TORCH_CHECK(playout->options->process_world_size() ==
                    playout->options->world_size(),
                "dynamics.fused-recon-riemann cubed-sphere symmetric-memory "
                "exchange currently requires one block rank per process");
    TORCH_CHECK(playout->has_process_group(),
                "dynamics.fused-recon-riemann cubed-sphere symmetric-memory "
                "exchange requires an initialized process group");
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
        dx1f, gas_constant, rho_grav);
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
        torch::Tensor());
  }
  if (u.size(1) > 1 && !options->disable_flux_x3()) {
    fused_recon_riemann_cuda(
        w, _flux3, /*dim=*/1, recon23_prim, recon23_vel, solver, eos,
        options->eos()->gammad(), options->eos()->density_floor(),
        options->eos()->pressure_floor(), options->eos()->limiter(),
        inv_mu_ratio_m1, cv_ratio_m1, u0, shallow_roe_dir_yz,
        FusedPrimitiveProjector::None, torch::Tensor(), torch::Tensor(), 0.,
        torch::Tensor());
  }

  if (cubed_sphere_layout) {
    bool exchange_dim2 = u.size(2) > 1 && !options->disable_flux_x2();
    bool exchange_dim3 = u.size(1) > 1 && !options->disable_flux_x3();
    if (exchange_dim2 || exchange_dim3) {
      auto cs_layout =
          dynamic_cast<CubedSphereLayoutImpl const*>(playout.get());
      TORCH_CHECK(cs_layout != nullptr,
                  "expected CubedSphereLayoutImpl for cubed-sphere layout");
      std::string group_name = "snapy:fused-recon-riemann:cubed-sphere";
      ensure_symmetric_group(*playout, group_name);
      int edge_len = std::max<int>(w.size(1), w.size(2));
      std::vector<int64_t> sizes = {4, w.size(0), edge_len, w.size(3)};
      std::vector<int64_t> strides = {w.size(0) * edge_len * w.size(3),
                                      edge_len * w.size(3), w.size(3), 1};
      auto symm_buffer = c10d::symmetric_memory::empty_strided_p2p(
          sizes, strides, w.scalar_type(), w.device(), std::nullopt,
          std::nullopt);
      auto symm = c10d::symmetric_memory::rendezvous(symm_buffer, group_name);
      clear_fused_signal_slots(symm, playout->comm);
      auto side_meta =
          make_side_meta(*cs_layout, exchange_dim2, exchange_dim3, w.device());
      int face = std::get<2>(cs_layout->loc_of(cs_layout->options->rank()));
      auto x2v = pmb->pcoord->x2v.to(w.options());
      auto x2f = pmb->pcoord->x2f.to(w.options());
      auto x3v = pmb->pcoord->x3v.to(w.options());
      auto x3f = pmb->pcoord->x3f.to(w.options());
      fused_cubed_sphere_exchange_cuda(
          w, _flux2, _flux3, symm_buffer, symm->get_buffer_ptrs_dev(),
          reinterpret_cast<uint32_t**>(symm->get_signal_pad_ptrs_dev()), face,
          symm->get_rank(), symm->get_world_size(), side_meta, x2v, x2f, x3v,
          x3f, recon23_prim, recon23_vel, solver, eos, options->eos()->gammad(),
          options->eos()->density_floor(), options->eos()->pressure_floor(),
          options->eos()->limiter(), inv_mu_ratio_m1, cv_ratio_m1, u0,
          shallow_roe_dir_yz, FusedPrimitiveProjector::None);
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
