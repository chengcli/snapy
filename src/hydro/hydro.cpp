// kintera
#include <kintera/utils/format.hpp>

// C/C++
#include <array>
#include <mutex>
#include <set>

// snap
#include <snap/snap.h>

#include <snap/layout/cubed_sphere_layout.hpp>
#include <snap/layout/layout.hpp>
#include <snap/mesh/meshblock.hpp>
#include <snap/utils/log.hpp>
#ifndef NOT_USE_NVSHMEM
#include <torch/csrc/distributed/c10d/symm_mem/SymmetricMemory.hpp>
#endif

#include "../eos/ideal_moist.hpp"
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

#ifndef NOT_USE_NVSHMEM
void ensure_symmetric_group(LayoutImpl const& layout,
                            std::string const& group_name) {
  static std::mutex mutex;
  static std::set<std::string> initialized;
  std::lock_guard<std::mutex> lock(mutex);
  if (initialized.count(group_name)) return;
  TORCH_CHECK(layout.comm != nullptr && layout.comm->store.defined(),
              "dynamics.fused-recon-riemann cubed-sphere exchange requires an "
              "initialized process-group store");
  c10d::symmetric_memory::set_backend("NVSHMEM");
  c10d::symmetric_memory::set_signal_pad_size(std::max<size_t>(
      1024, 4 * layout.options->process_world_size() * sizeof(uint32_t)));
  c10d::symmetric_memory::set_group_info(
      group_name, layout.options->process_rank(),
      layout.options->process_world_size(), layout.comm->store);
  initialized.insert(group_name);
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
#endif

}  // namespace

HydroImpl::HydroImpl(const HydroOptions& options_, torch::nn::Module* p)
    : options(options_) {
  pmb = dynamic_cast<MeshBlockImpl const*>(p);
  reset();
}

void HydroImpl::reset() {
  TORCH_CHECK(pmb, "[Hydro] Parent MeshBlock is null");

  //// ---- (1) set up equation-of-state model ---- ////
  peos = EquationOfStateImpl::create(options->eos(), this);
  if (options->verbose()) {
    SINFO(Hydro) << "EOS type: " << peos->options->type() << "\n";
  }

  //// ---- (2) set up primitive projector model ---- ////
  if (options->proj() != nullptr) {
    pproj = PrimitiveProjectorImpl::create(options->proj(), this);

    if (options->verbose()) {
      SINFO(Hydro) << "Primitive projector type: " << pproj->options->type()
                   << "\n";
    }
  }

  //// ---- (3) set up reconstruction-x1 model ---- ////
  precon1 = ReconstructImpl::create(options->recon1(), this, "recon1");
  if (options->verbose()) {
    SINFO(Hydro) << "Reconstruction-x1 type: "
                 << precon1->pinterp1->options->type() << "\n";
  }

  //// ---- (4) set up reconstruction-x23 model ---- ////
  precon23 = ReconstructImpl::create(options->recon23(), this, "recon23");
  if (options->verbose()) {
    SINFO(Hydro) << "Reconstruction-x2/x3 type: "
                 << precon23->pinterp1->options->type() << "\n";
  }

  //// ---- (5) set up riemann-solver model ---- ////
  priemann = RiemannSolverImpl::create(options->riemann(), this);
  if (options->verbose()) {
    SINFO(Hydro) << "Riemann solver type: " << priemann->options->type()
                 << "\n";
  }

  //// ---- (6) set up implicit solver ---- ////
  if (options->icorr()) {
    picorr = ImplicitHydroImpl::create(options->icorr(), this);
    if (options->verbose()) {
      SINFO(Hydro) << "Implicit correction type: " << picorr->options->type()
                   << "\n";
    }
  }

  //// ---- (7) set up sedimentation ---- ////
  if (options->sed() != nullptr) {
    psed = SedHydroImpl::create(options->sed(), this);
    if (options->verbose()) {
      SINFO(Hydro) << "Sedimentation particle ids: "
                   << fmt::format("{}", psed->options->sedvel()->particle_ids())
                   << "\n";
    }
  }

  //// ---- (8) set up forcings ---- ////
  auto forcing_names = _register_forcings_module();
  if (options->verbose()) {
    SINFO(Hydro) << "Forcings: " << fmt::format("{}", forcing_names) << "\n";
  }

  //// ---- (9) register all forcings ---- ////
  for (auto i = 0; i < forcings.size(); i++) {
    register_module(forcing_names[i], forcings[i].ptr());
  }

  //// ---- (10) populate buffers ---- ////
  int nc1 = pmb->options->coord()->nc1();
  int nc2 = pmb->options->coord()->nc2();
  int nc3 = pmb->options->coord()->nc3();
  int nvar = peos->nvar();

  if (nc1 > 1) {
    _flux1 = register_buffer(
        "F1", torch::zeros({nvar, nc3, nc2, nc1}, torch::kFloat64));
  } else {
    _flux1 = register_buffer("F1", torch::Tensor());
  }

  if (nc2 > 1) {
    _flux2 = register_buffer(
        "F2", torch::zeros({nvar, nc3, nc2, nc1}, torch::kFloat64));
  } else {
    _flux2 = register_buffer("F2", torch::Tensor());
  }

  if (nc3 > 1) {
    _flux3 = register_buffer(
        "F3", torch::zeros({nvar, nc3, nc2, nc1}, torch::kFloat64));
  } else {
    _flux3 = register_buffer("F3", torch::Tensor());
  }

  _div = register_buffer("D",
                         torch::zeros({nvar, nc3, nc2, nc1}, torch::kFloat64));

  _imp = register_buffer("M",
                         torch::zeros({nvar, nc3, nc2, nc1}, torch::kFloat64));
}

double HydroImpl::max_time_step(torch::Tensor w, torch::Tensor solid) const {
  auto sub3 = pmb->part({0, 0, 0}, PartOptions().exterior(false).ndim(3));

  torch::Tensor cs;
  if (options->eos()->type() == "aneos") {
    cs = peos->compute("W->L", {w});
  } else {
    auto gamma = peos->compute("W->A", {w});
    cs = peos->compute("WA->L", {w, gamma});
  }

  if (solid.defined()) {
    cs = torch::where(solid, 1.e-8, cs);
  }

  auto dt_min = torch::tensor({1.e9, 1.e9, 1.e9},
                              torch::dtype(torch::kFloat64).device(w.device()));

  auto icorr = options->icorr();

  if (icorr) {
    if ((cs.size(2) > 1) &&
        (!(icorr->scheme() & 1) || (cs.size(0) == 1 && cs.size(1) == 1))) {
      dt_min[0] = (pmb->pcoord->center_width1() / (w[IVX].abs() + cs))
                      .index(sub3)
                      .min();
    }

    if ((cs.size(1) > 1) && (!((icorr->scheme() >> 1) & 1))) {
      dt_min[1] = (pmb->pcoord->center_width2() / (w[IVY].abs() + cs))
                      .index(sub3)
                      .min();
    }

    if ((cs.size(0) > 1) && (!((icorr->scheme() >> 2) & 1))) {
      dt_min[2] = (pmb->pcoord->center_width3() / (w[IVZ].abs() + cs))
                      .index(sub3)
                      .min();
    }
  } else {
    if (cs.size(2) > 1) {
      dt_min[0] = (pmb->pcoord->center_width1() / (w[IVX].abs() + cs))
                      .index(sub3)
                      .min();
    }

    if (cs.size(1) > 1) {
      dt_min[1] = (pmb->pcoord->center_width2() / (w[IVY].abs() + cs))
                      .index(sub3)
                      .min();
    }

    if (cs.size(0) > 1) {
      dt_min[2] = (pmb->pcoord->center_width3() / (w[IVZ].abs() + cs))
                      .index(sub3)
                      .min();
    }
  }

  double dt = dt_min.min().item<double>();
  if (pdiffusion) dt = std::min(dt, pdiffusion->max_time_step(w));
  return dt;
}

torch::Tensor HydroImpl::forward(double dt, torch::Tensor u,
                                 Variables const& other) {
  if (options->fused_recon_riemann()) {
#ifdef NOT_USE_NVSHMEM
    return _forward_staged(dt, u, other);
#else
    return _forward_fused_recon_riemann(dt, u, other);
#endif
  }
  return _forward_staged(dt, u, other);
}

torch::Tensor HydroImpl::_forward_staged(double dt, torch::Tensor u,
                                         Variables const& other) {
  enum { DIM1 = 3, DIM2 = 2, DIM3 = 1 };
  bool has_solid = other.count("solid");
  auto start = std::chrono::high_resolution_clock::now();

  auto playout = pmb->get_layout();

  //// ------------ (1) Calculate Primitives ------------ ////
  auto const& w = other.at("hydro_w");

  peos->forward(u, w);
  if (options->verbose()) {
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    SINFO(Hydro) << "EOS time (s): " << elapsed.count() << "\n";
    start = std::chrono::high_resolution_clock::now();
  }

  if (has_solid) {
    pmb->pib->mark_prim_solid_(w, other.at("solid"));
  }

  // hydrostatic pressure correction
  torch::Tensor rho_grav = torch::zeros_like(w[IDN]);

  //// ------------ (2) Calculate dimension 1 flux ------------ ////
  if (u.size(DIM1) > 1) {
    torch::Tensor wtmp;
    if (pproj) {
      auto wp = pproj->forward(w, pmb->pcoord->dx1f);
      wtmp = precon1->forward(wp, DIM1);
      rho_grav = pproj->restore_inplace(wtmp);
    } else {
      wtmp = precon1->forward(w, DIM1);
    }

    auto wlr1 =
        has_solid ? pmb->pib->forward(wtmp, DIM1, other.at("solid")) : wtmp;

    if (!options->disable_flux_x1()) {
      priemann->forward(wlr1[ILT], wlr1[IRT], DIM1, _flux1);
      if (options->verbose()) {
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end - start;
        SINFO(Hydro) << "Flux-x1 time (s): " << elapsed.count() << "\n";
        start = std::chrono::high_resolution_clock::now();
      }
    }

    // add sedimentation flux
    if (psed) psed->forward(w, _flux1);
  }

  //// ------------ (3.A) Calculate dimension 2 LR states ------------ ////
  torch::Tensor wtmp2, wtmp3;
  SyncOptions sync_opts;
  sync_opts.cross_panel_only(true).interpolate(false).type(kPrimitive);
  std::vector<CommWorkPtr> works;
  Variables send_vars2, send_vars3;

  if (u.size(DIM2) > 1) {
    wtmp2 = precon23->forward(w, DIM2);

    // sync left/right states across faces for cubed sphere layout
    if (playout->options->type() == "cubed-sphere") {
      send_vars2["hydro_wl:+"] = wtmp2[ILT];
      send_vars2["hydro_wr:-"] = wtmp2[IRT];
      pmb->begin_exchange(send_vars2, sync_opts.dim(DIM2));
    }
  }

  //// ------------ (3.B) Calculate dimension 3 LR states ------------ ////
  if (u.size(DIM3) > 1) {
    wtmp3 = precon23->forward(w, DIM3);

    // sync left/right states across faces for cubed sphere layout
    if (playout->options->type() == "cubed-sphere") {
      send_vars3["hydro_wl:+"] = wtmp3[ILT];
      send_vars3["hydro_wr:-"] = wtmp3[IRT];
      pmb->begin_exchange(send_vars3, sync_opts.dim(DIM3));
    }
  }

  if (playout->options->type() == "cubed-sphere") {
    bool exchange_dim2 = u.size(DIM2) > 1;
    bool exchange_dim3 = u.size(DIM3) > 1;
    if (exchange_dim2) {
      pmb->launch_exchange(sync_opts.dim(DIM2), works);
    }
    if (exchange_dim3) {
      pmb->launch_exchange(sync_opts.dim(DIM3), works);
    }
    if (exchange_dim2) {
      pmb->finalize_exchange(send_vars2, sync_opts.dim(DIM2), works);
    }
    if (exchange_dim3) {
      pmb->finalize_exchange(send_vars3, sync_opts.dim(DIM3), works);
    }
  }

  //// ------------ (4.A) Calculate dimension 2 flux ------------ ////
  if (u.size(DIM2) > 1) {
    auto wlr2 =
        has_solid ? pmb->pib->forward(wtmp2, DIM2, other.at("solid")) : wtmp2;
    if (!options->disable_flux_x2()) {
      priemann->forward(wlr2[ILT], wlr2[IRT], DIM2, _flux2);
      if (options->verbose()) {
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end - start;
        SINFO(Hydro) << "Flux-x2 time (s): " << elapsed.count() << "\n";
        start = std::chrono::high_resolution_clock::now();
      }
    }
  }

  //// ------------ (4.B) Calculate dimension 3 flux ------------ ////
  if (u.size(DIM3) > 1) {
    auto wlr3 =
        has_solid ? pmb->pib->forward(wtmp3, DIM3, other.at("solid")) : wtmp3;
    if (!options->disable_flux_x3()) {
      priemann->forward(wlr3[ILT], wlr3[IRT], DIM3, _flux3);
      if (options->verbose()) {
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end - start;
        SINFO(Hydro) << "Flux-x3 time (s): " << elapsed.count() << "\n";
        start = std::chrono::high_resolution_clock::now();
      }
    }
  }

  //// ------------ (5) Calculate flux divergence ------------ ////
  _div.set_(pmb->pcoord->forward(w, _flux1, _flux2, _flux3));
  if (options->verbose()) {
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    SINFO(Hydro) << "Divergence time (s): " << elapsed.count() << "\n";
    start = std::chrono::high_resolution_clock::now();
  }

  //// ------------ (6) Calculate external forcing ------------ ////
  auto du = torch::zeros_like(_div);
  auto interior = pmb->part({0, 0, 0}, PartOptions().exterior(false));
  du.index(interior) = -dt * _div.index(interior);

  auto temp = peos->compute("W->T", {w});
  for (auto& f : forcings) f.forward(du, w, temp, dt);

  // apply hydrostatic correction
  if (options->grav() && (options->grav()->non_hydrostatic() < 1.)) {
    du[IVX] += dt * rho_grav * (1. - options->grav()->non_hydrostatic());
    du[IPR] +=
        dt * w[IVX] * rho_grav * (1. - options->grav()->non_hydrostatic());
  }

  if (options->verbose()) {
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    SINFO(Hydro) << "Forcing time (s): " << elapsed.count() << "\n";
    start = std::chrono::high_resolution_clock::now();
  }

  //// ------------ (7) Perform implicit correction ------------ ////
  if (picorr) {
    torch::Tensor wi;
    if (has_solid) {
      wi = torch::where(other.at("solid").unsqueeze(0).expand_as(w),
                        other.at("fill_solid_hydro_w"), w);
      du.masked_fill_(other.at("solid").unsqueeze(0).expand_as(du), 0.0);
    } else {
      wi = w;
    }

    auto du0 = du.clone();
    du[IPR].sub_(peos->internal_energy_offset(du));

    torch::Tensor gamma;
    if (options->eos()->type() == "aneos") {
      auto cs = peos->compute("W->L", {w});
      gamma = peos->compute("WL->A", {w, cs});
    } else {
      gamma = peos->compute("W->A", {wi});
    }
    picorr->forward(du, wi, gamma, dt);
    du[IPR].add_(peos->internal_energy_offset(du));

    _imp.copy_(du);
    _imp.sub_(du0);

    if (options->verbose()) {
      auto end = std::chrono::high_resolution_clock::now();
      std::chrono::duration<double> elapsed = end - start;
      SINFO(Hydro) << "Implicit time (s): " << elapsed.count() << "\n";
      start = std::chrono::high_resolution_clock::now();
    }
  }

  return du;
}

torch::Tensor HydroImpl::_forward_fused_recon_riemann(double dt,
                                                      torch::Tensor u,
                                                      Variables const& other) {
#ifdef NOT_USE_NVSHMEM
  TORCH_CHECK(false,
              "dynamics.fused-recon-riemann is disabled unless snapy is built "
              "with NVSHMEM=ON");
#else
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
  TORCH_CHECK(!pproj,
              "dynamics.fused-recon-riemann does not yet support primitive "
              "projectors");
  TORCH_CHECK(!psed,
              "dynamics.fused-recon-riemann does not yet support "
              "sedimentation fluxes");

  auto eos_type = options->eos()->type();
  TORCH_CHECK(eos_type == "ideal-gas" || eos_type == "ideal-moist",
              "dynamics.fused-recon-riemann supports EOS types ideal-gas and "
              "ideal-moist, but got ",
              eos_type);

  auto riemann_type = options->riemann()->type();
  TORCH_CHECK(riemann_type == "lmars" || riemann_type == "hllc",
              "dynamics.fused-recon-riemann supports Riemann solvers lmars "
              "and hllc, but got ",
              riemann_type);

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
  auto solver = riemann_type == "lmars" ? FusedRiemannSolver::LMARS
                                        : FusedRiemannSolver::HLLC;
  auto eos =
      eos_type == "ideal-gas" ? FusedEos::IdealGas : FusedEos::IdealMoist;

  torch::Tensor inv_mu_ratio_m1, cv_ratio_m1, u0;
  if (eos == FusedEos::IdealMoist) {
    auto ideal_moist = dynamic_cast<IdealMoistImpl const*>(peos.get());
    TORCH_CHECK(ideal_moist != nullptr,
                "dynamics.fused-recon-riemann expected IdealMoistImpl");
    inv_mu_ratio_m1 = ideal_moist->inv_mu_ratio_m1.to(w.options());
    cv_ratio_m1 = ideal_moist->cv_ratio_m1.to(w.options());
    u0 = ideal_moist->u0.to(w.options());
  }

  if (u.size(3) > 1 && !options->disable_flux_x1()) {
    fused_recon_riemann_cuda(
        w, _flux1, /*dim=*/3, recon1_prim, recon1_vel, solver, eos,
        options->eos()->gammad(), options->eos()->density_floor(),
        options->eos()->pressure_floor(), options->eos()->limiter(),
        inv_mu_ratio_m1, cv_ratio_m1, u0);
  }
  if (u.size(2) > 1 && !options->disable_flux_x2()) {
    fused_recon_riemann_cuda(
        w, _flux2, /*dim=*/2, recon23_prim, recon23_vel, solver, eos,
        options->eos()->gammad(), options->eos()->density_floor(),
        options->eos()->pressure_floor(), options->eos()->limiter(),
        inv_mu_ratio_m1, cv_ratio_m1, u0);
  }
  if (u.size(1) > 1 && !options->disable_flux_x3()) {
    fused_recon_riemann_cuda(
        w, _flux3, /*dim=*/1, recon23_prim, recon23_vel, solver, eos,
        options->eos()->gammad(), options->eos()->density_floor(),
        options->eos()->pressure_floor(), options->eos()->limiter(),
        inv_mu_ratio_m1, cv_ratio_m1, u0);
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
          sizes, strides, w.scalar_type(), w.device(), group_name,
          std::nullopt);
      auto symm = c10d::symmetric_memory::rendezvous(symm_buffer, group_name);
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
          options->eos()->limiter(), inv_mu_ratio_m1, cv_ratio_m1, u0);
    }
  }

  _div.set_(pmb->pcoord->forward(w, _flux1, _flux2, _flux3));

  auto du = torch::zeros_like(_div);
  auto interior = pmb->part({0, 0, 0}, PartOptions().exterior(false));
  du.index(interior) = -dt * _div.index(interior);

  auto temp = peos->compute("W->T", {w});
  for (auto& f : forcings) f.forward(du, w, temp, dt);
  return du;
#endif
}

torch::Tensor HydroImpl::implicit_mass_correction() const {
  return picorr ? picorr->mass_correction() : torch::Tensor();
}

std::shared_ptr<HydroImpl> HydroImpl::create(HydroOptions const& opts,
                                             torch::nn::Module* p,
                                             std::string const& name) {
  TORCH_CHECK(p, "[Hydro] Parent module is null");
  TORCH_CHECK(opts, "[Hydro] Options pointer is null");

  return p->register_module(name, Hydro(opts, p));
}

/*void check_recon(torch::Tensor wlr, int nghost, int extend_x1, int extend_x2,
                 int extend_x3) {
  auto interior =
      get_interior(wlr.sizes(), nghost, extend_x1, extend_x2, extend_x3);

  int dim = extend_x1 == 1 ? 1 : (extend_x2 == 1 ? 2 : 3);
  TORCH_CHECK(wlr.index(interior).select(1, IDN).min().item<double>() > 0.,
              "Negative density detected after reconstruction in dimension ",
              dim);
  TORCH_CHECK(wlr.index(interior).select(1, IPR).min().item<double>() > 0.,
              "Negative pressure detected after reconstruction in dimension ",
              dim);
}

void check_eos(torch::Tensor w, int nghost) {
  auto interior = get_interior(w.sizes(), nghost);
  TORCH_CHECK(w.index(interior)[IDN].min().item<double>() > 0.,
              "Negative density detected after EOS. ",
              "Suggestions: 1) Reducting the CFL number;",
              " 2) Activate EOS limiter and set the density floor");
  TORCH_CHECK(w.index(interior)[IPR].min().item<double>() > 0.,
              "Negative pressure detected after EOS. ",
              "Suggestions: 1) Reducting the CFL number; ",
              " 2) Activate EOS limiter and set the pressure floor");
}*/

}  // namespace snap
