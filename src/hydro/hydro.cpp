// kintera
#include <kintera/utils/format.hpp>

// snap
#include <snap/snap.h>

#include <snap/layout/layout.hpp>
#include <snap/mesh/meshblock.hpp>

#include "hydro.hpp"

namespace snap {

HydroImpl::HydroImpl(const HydroOptions& options_, torch::nn::Module* p)
    : options(options_) {
  if (p) {
    parent =
        std::dynamic_pointer_cast<MeshBlockImpl const>(p->shared_from_this());
  }
  reset();
}

void HydroImpl::reset() {
  int rank = get_rank();

  //// ---- (1) set up coordinate model ---- ////
  pcoord = CoordinateImpl::create(options->coord(), this);
  if (options->verbose() && rank == 0) {
    std::cout << "[Hydro] Coordinate type: " << pcoord->options->type() << "\n";
  }

  //// ---- (2) set up equation-of-state model ---- ////
  peos = EquationOfStateImpl::create(options->eos(), this);
  if (options->verbose() && rank == 0) {
    std::cout << "[Hydro] EOS type: " << peos->options->type() << "\n";
  }

  //// ---- (3) set up primitive projector model ---- ////
  if (options->proj() != nullptr) {
    pproj = PrimitiveProjectorImpl::create(options->proj(), this);

    if (options->verbose() && rank == 0) {
      std::cout << "[Hydro] Primitive projector type: "
                << pproj->options->type() << "\n";
    }
  }

  //// ---- (4) set up reconstruction-x1 model ---- ////
  precon1 = ReconstructImpl::create(options->recon1(), this, "recon1");
  if (options->verbose() && rank == 0) {
    std::cout << "[Hydro] Reconstruction-x1 type: "
              << precon1->pinterp1->options->type() << "\n";
  }

  //// ---- (5) set up reconstruction-x23 model ---- ////
  precon23 = ReconstructImpl::create(options->recon23(), this, "recon23");
  if (options->verbose() && rank == 0) {
    std::cout << "[Hydro] Reconstruction-x2/x3 type: "
              << precon23->pinterp1->options->type() << "\n";
  }

  //// ---- (6) set up riemann-solver model ---- ////
  priemann = RiemannSolverImpl::create(options->riemann(), this);
  if (options->verbose() && rank == 0) {
    std::cout << "[Hydro] Riemann solver type: " << priemann->options->type()
              << "\n";
  }

  //// ---- (7) set up internal boundary ---- ////
  pib = InternalBoundaryImpl::create(options->ib(), this);
  if (options->verbose() && rank == 0) {
    std::cout << "[Hydro] Internal boundary max-iter: "
              << pib->options->max_iter() << "\n";
  }

  //// ---- (8) set up implicit solver ---- ////
  if (options->icorr()) {
    picorr = ImplicitCorrectionImpl::create(options->icorr(), this);
    if (options->verbose() && rank == 0) {
      std::cout << "[Hydro] Implicit correction type: "
                << picorr->options->type() << "\n";
    }
  }

  //// ---- (9) set up sedimentation ---- ////
  if (options->sed() != nullptr) {
    psed = SedHydroImpl::create(options->sed(), this);
    if (options->verbose() && rank == 0) {
      std::cout << "[Hydro] Sedimentation particle ids: "
                << fmt::format("{}", psed->options->sedvel()->particle_ids())
                << "\n";
    }
  }

  //// ---- (10) set up forcings ---- ////
  auto forcing_names = register_forcings_module();
  if (options->verbose() && rank == 0) {
    std::cout << "[Hydro] Forcings: " << fmt::format("{}", forcing_names)
              << "\n";
  }

  //// ---- (11) register all forcings ---- ////
  for (auto i = 0; i < forcings.size(); i++) {
    register_module(forcing_names[i], forcings[i].ptr());
  }

  //// ---- (12) populate buffers ---- ////
  int nc1 = options->coord()->nc1();
  int nc2 = options->coord()->nc2();
  int nc3 = options->coord()->nc3();
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
  torch::Tensor cs;
  if (options->eos()->type() == "aneos" ||
      options->eos()->type() == "plume-eos") {
    cs = peos->compute("W->L", {w});
  } else {
    auto gamma = peos->compute("W->A", {w});
    cs = peos->compute("WA->L", {w, gamma});
  }

  if (solid.defined()) {
    cs = torch::where(solid, 1.e-8, cs);
  }

  double dt1 = 1.e9, dt2 = 1.e9, dt3 = 1.e9;
  auto icorr = options->icorr();

  if (icorr) {
    if ((cs.size(2) > 1) &&
        (!(icorr->scheme() & 1) || (cs.size(0) == 1 && cs.size(1) == 1))) {
      dt1 = torch::min(pcoord->center_width1() / (w[IVX].abs() + cs))
                .item<double>();
    }

    if ((cs.size(1) > 1) && (!((icorr->scheme() >> 1) & 1))) {
      dt2 = torch::min(pcoord->center_width2() / (w[IVY].abs() + cs))
                .item<double>();
    }

    if ((cs.size(0) > 1) && (!((icorr->scheme() >> 2) & 1))) {
      dt3 = torch::min(pcoord->center_width3() / (w[IVZ].abs() + cs))
                .item<double>();
    }
  } else {
    if (cs.size(2) > 1) {
      dt1 = torch::min(pcoord->center_width1() / (w[IVX].abs() + cs))
                .item<double>();
    }

    if (cs.size(1) > 1) {
      dt2 = torch::min(pcoord->center_width2() / (w[IVY].abs() + cs))
                .item<double>();
    }

    if (cs.size(0) > 1) {
      dt3 = torch::min(pcoord->center_width3() / (w[IVZ].abs() + cs))
                .item<double>();
    }
  }

  return std::min({dt1, dt2, dt3});
}

torch::Tensor HydroImpl::forward(double dt, torch::Tensor u,
                                 Variables const& other) {
  auto pmb = parent.lock();

  enum { DIM1 = 3, DIM2 = 2, DIM3 = 1 };
  bool has_solid = other.count("solid");
  auto start = std::chrono::high_resolution_clock::now();

  //// ------------ (1) Calculate Primitives ------------ ////
  auto const& w = other.at("hydro_w");

  peos->forward(u, w);
  if (options->verbose()) {
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    std::cout << "[Hydro] EOS time (s): " << elapsed.count() << "\n";
    start = std::chrono::high_resolution_clock::now();
  }

  if (has_solid) {
    pib->mark_prim_solid_(w, other.at("solid"));
  }

  auto playout = MeshBlockImpl::get_layout();

  if (playout->options->type() == "cubed-sphere") {
    TORCH_CHECK(pmb, "[Hydro] Parent MeshBlock is null");
  }

  //// ------------ (2) Calculate dimension 1 flux ------------ ////
  if (u.size(DIM1) > 1) {
    torch::Tensor wtmp;
    if (pproj) {
      auto wp = pproj->forward(w, pcoord->dx1f);
      wtmp = precon1->forward(wp, DIM1);
      pproj->restore_inplace(wtmp);
    } else {
      wtmp = precon1->forward(w, DIM1);
    }

    auto wlr1 = has_solid ? pib->forward(wtmp, DIM1, other.at("solid")) : wtmp;

    if (!options->disable_flux_x1()) {
      priemann->forward(wlr1[ILT], wlr1[IRT], DIM1, _flux1);
      if (options->verbose()) {
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end - start;
        std::cout << "[Hydro] Flux-x1 time (s): " << elapsed.count() << "\n";
        start = std::chrono::high_resolution_clock::now();
      }
    }

    // add sedimentation flux
    if (psed) psed->forward(w, _flux1);
  }

  //// ------------ (3) Calculate dimension 2 flux ------------ ////
  if (u.size(DIM2) > 1) {
    auto wtmp = precon23->forward(w, DIM2);

    // sync left/right states across faces for cubed sphere layout
    if (playout->options->type() == "cubed-sphere") {
      SyncOptions sync_opts;
      sync_opts.cross_panel_only(true);
      sync_opts.dim(DIM2);
      sync_opts.interpolate(false);
      sync_opts.type(kPrimitive);

      Variables sync_vars;
      sync_vars["hydro_wl"] = wtmp[ILT];
      sync_vars["hydro_wr"] = wtmp[IRT];
      playout->forward(pmb.get(), sync_vars, sync_opts);
    }

    auto wlr2 = has_solid ? pib->forward(wtmp, DIM2, other.at("solid")) : wtmp;
    if (!options->disable_flux_x2()) {
      priemann->forward(wlr2[ILT], wlr2[IRT], DIM2, _flux2);
      if (options->verbose()) {
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end - start;
        std::cout << "[Hydro] Flux-x2 time (s): " << elapsed.count() << "\n";
        start = std::chrono::high_resolution_clock::now();
      }
    }
  }

  //// ------------ (4) Calculate dimension 3 flux ------------ ////
  if (u.size(DIM3) > 1) {
    auto wtmp = precon23->forward(w, DIM3);

    // sync left/right states across faces for cubed sphere layout
    if (playout->options->type() == "cubed-sphere") {
      SyncOptions sync_opts;
      sync_opts.cross_panel_only(true);
      sync_opts.dim(DIM3);
      sync_opts.interpolate(false);
      sync_opts.type(kPrimitive);

      Variables sync_vars;
      sync_vars["hydro_wl"] = wtmp[ILT];
      sync_vars["hydro_wr"] = wtmp[IRT];
      playout->forward(pmb.get(), sync_vars, sync_opts);
    }

    auto wlr3 = has_solid ? pib->forward(wtmp, DIM3, other.at("solid")) : wtmp;
    if (!options->disable_flux_x3()) {
      priemann->forward(wlr3[ILT], wlr3[IRT], DIM3, _flux3);
      if (options->verbose()) {
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end - start;
        std::cout << "[Hydro] Flux-x3 time (s): " << elapsed.count() << "\n";
        start = std::chrono::high_resolution_clock::now();
      }
    }
  }

  //// ------------ (5) Calculate flux divergence ------------ ////
  _div.set_(pcoord->forward(w, _flux1, _flux2, _flux3));

  //// ------------ (6) Calculate external forcing ------------ ////
  auto du = -dt * _div;
  auto temp = peos->compute("W->T", {w});
  for (auto& f : forcings) f.forward(du, w, temp, dt);
  if (options->verbose()) {
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    std::cout << "[Hydro] Forcing time (s): " << elapsed.count() << "\n";
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

    torch::Tensor gamma;
    if (options->eos()->type() == "aneos") {
      auto cs = peos->compute("W->L", {w});
      gamma = peos->compute("WL->A", {w, cs});
    } else {
      gamma = peos->compute("W->A", {wi});
    }
    _imp.set_(picorr->forward(du, wi, gamma, dt));

    if (options->verbose()) {
      auto end = std::chrono::high_resolution_clock::now();
      std::chrono::duration<double> elapsed = end - start;
      std::cout << "[Hydro] Implicit time (s): " << elapsed.count() << "\n";
      start = std::chrono::high_resolution_clock::now();
    }
  }

  return du;
}

std::shared_ptr<HydroImpl> HydroImpl::create(HydroOptions const& opts,
                                             torch::nn::Module* p,
                                             std::string const& name) {
  TORCH_CHECK(p, "[Hydro] Parent module is null");
  TORCH_CHECK(opts, "[Hydro] Options pointer is null");

  return p->register_module(name, Hydro(opts, p));
}

void check_recon(torch::Tensor wlr, int nghost, int extend_x1, int extend_x2,
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
}

}  // namespace snap
