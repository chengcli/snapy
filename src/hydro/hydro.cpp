// base
#include <configure.h>

// snap
#include <snap/snap.h>

#include <snap/registry.hpp>

#include "hydro.hpp"

namespace snap {
HydroImpl::HydroImpl(const HydroOptions& options_) : options(options_) {
  reset();
}

void HydroImpl::reset() {
  // set up coordinate model
  pcoord = register_module_op(this, "coord", options.coord());
  options.coord() = pcoord->options;

  // set up equation-of-state model
  peos = register_module_op(this, "eos", options.eos());
  options.eos() = peos->options;

  // set up primitive projector model
  pproj = register_module("proj", PrimitiveProjector(options.proj()));
  options.proj() = pproj->options;

  // set up reconstruction-x1 model
  precon1 = register_module("recon1", Reconstruct(options.recon1()));
  options.recon1() = precon1->options;

  // set up reconstruction-x23 model
  precon23 = register_module("recon23", Reconstruct(options.recon23()));
  options.recon23() = precon23->options;

  // set up riemann-solver model
  priemann = register_module_op(this, "riemann", options.riemann());
  options.riemann() = priemann->options;

  // set up internal boundary
  pib = register_module("ib", InternalBoundary(options.ib()));
  options.ib() = pib->options;

  // set up vertical implicit solver
  pvic = register_module("vic", VerticalImplicit(options.vic()));
  options.vic() = pvic->options;

  // set up forcings
  if (options.grav().grav1() != 0.0 || options.grav().grav2() != 0.0 ||
      options.grav().grav3() != 0.0) {
    forcings.push_back(torch::nn::AnyModule(ConstGravity(options.grav())));
  }

  if (options.coriolis().omega1() != 0.0 ||
      options.coriolis().omega2() != 0.0 ||
      options.coriolis().omega3() != 0.0) {
    forcings.push_back(torch::nn::AnyModule(Coriolis123(options.coriolis())));
  }

  if (options.coriolis().omegax() != 0.0 ||
      options.coriolis().omegay() != 0.0 ||
      options.coriolis().omegaz() != 0.0) {
    forcings.push_back(torch::nn::AnyModule(CoriolisXYZ(options.coriolis())));
  }

  // register all forcings
  for (auto i = 0; i < forcings.size(); i++) {
    register_module("forcing" + std::to_string(i), forcings[i].ptr());
  }

  // populate buffers
  int nc1 = options.coord().nc1();
  int nc2 = options.coord().nc2();
  int nc3 = options.coord().nc3();
  int nvar = peos->nvar();

  if (nc1 > 1) {
    _flux1 = register_buffer(
        "F1", torch::empty({nvar, nc3, nc2, nc1}, torch::kFloat64));
  } else {
    _flux1 = register_buffer("F1", torch::Tensor());
  }

  if (nc2 > 1) {
    _flux2 = register_buffer(
        "F2", torch::empty({nvar, nc3, nc2, nc1}, torch::kFloat64));
  } else {
    _flux2 = register_buffer("F2", torch::Tensor());
  }

  if (nc3 > 1) {
    _flux3 = register_buffer(
        "F3", torch::empty({nvar, nc3, nc2, nc1}, torch::kFloat64));
  } else {
    _flux3 = register_buffer("F3", torch::Tensor());
  }

  _div = register_buffer("D",
                         torch::empty({nvar, nc3, nc2, nc1}, torch::kFloat64));

  _vic = register_buffer("I",
                         torch::empty({nvar, nc3, nc2, nc1}, torch::kFloat64));

  // initialize timers
  timer["U->W"] = 0.0;
  timer["W->LR1"] = 0.0;
  timer["LR1->F1"] = 0.0;
  timer["W->LR2"] = 0.0;
  timer["LR2->F2"] = 0.0;
  timer["W->LR3"] = 0.0;
  timer["LR3->F3"] = 0.0;
  timer["F->D"] = 0.0;
  timer["W->R"] = 0.0;
  timer["U->I"] = 0.0;
}

double HydroImpl::max_time_step(torch::Tensor w, torch::Tensor solid) const {
  auto gamma = peos->compute("W->A", {w});
  auto cs = peos->compute("WA->L", {w, gamma});

  if (solid.defined()) {
    cs = torch::where(solid, 1.e-8, cs);
  }

  double dt1 = 1.e9, dt2 = 1.e9, dt3 = 1.e9;

  if ((cs.size(2) > 1) && (pvic->options.scheme() == 0)) {
    dt1 = torch::min(pcoord->center_width1() / (w[Index::IVX].abs() + cs))
              .item<double>();
  }

  if (cs.size(1) > 1) {
    dt2 = torch::min(pcoord->center_width2() / (w[Index::IVY].abs() + cs))
              .item<double>();
  }

  if (cs.size(0) > 1) {
    dt3 = torch::min(pcoord->center_width3() / (w[Index::IVZ].abs() + cs))
              .item<double>();
  }

  return std::min({dt1, dt2, dt3});
}

torch::Tensor HydroImpl::forward(torch::Tensor u, double dt,
                                 torch::Tensor solid) {
  enum { DIM1 = 3, DIM2 = 2, DIM3 = 1 };
  auto const& w = peos->get_buffer("W");

  auto start = std::chrono::high_resolution_clock::now();
  //// ------------ (1) Calculate Primitives ------------ ////
  w.set_(pib->mark_solid(peos->forward(u), solid));

  auto time1 = std::chrono::high_resolution_clock::now();
  timer["U->W"] +=
      std::chrono::duration<double, std::milli>(time1 - start).count();

  //// ------------ (2) Calculate dimension 1 flux ------------ ////
  std::chrono::high_resolution_clock::time_point time2;

  if (u.size(DIM1) > 1) {
    auto wp = pproj->forward(w, pcoord->dx1f);
    auto wtmp = precon1->forward(wp, DIM1);

    auto time2a = std::chrono::high_resolution_clock::now();
    timer["W->LR1"] +=
        std::chrono::duration<double, std::milli>(time2a - time1).count();

    pproj->restore_inplace(wtmp);
    auto wlr1 = pib->forward(wtmp, DIM1, solid);

    priemann->forward(wlr1[Index::ILT], wlr1[Index::IRT], DIM1, _flux1);

    time2 = std::chrono::high_resolution_clock::now();
    timer["LR1->F1"] +=
        std::chrono::duration<double, std::milli>(time2 - time2a).count();
  }

  //// ------------ (3) Calculate dimension 2 flux ------------ ////
  if (u.size(DIM2) > 1) {
    auto wtmp = precon23->forward(w, DIM2);

    auto time2c = std::chrono::high_resolution_clock::now();
    timer["W->LR2"] +=
        std::chrono::duration<double, std::milli>(time2c - time2).count();

    auto wlr2 = pib->forward(wtmp, DIM2, solid);
    priemann->forward(wlr2[Index::ILT], wlr2[Index::IRT], DIM2, _flux2);

    time2 = std::chrono::high_resolution_clock::now();
    timer["LR2->F2"] +=
        std::chrono::duration<double, std::milli>(time2 - time2c).count();
  }

  //// ------------ (4) Calculate dimension 3 flux ------------ ////
  if (u.size(DIM3) > 1) {
    auto wtmp = precon23->forward(w, DIM3);

    auto time2e = std::chrono::high_resolution_clock::now();
    timer["W->LR3"] +=
        std::chrono::duration<double, std::milli>(time2e - time2).count();

    auto wlr3 = pib->forward(wtmp, DIM3, solid);
    priemann->forward(wlr3[Index::ILT], wlr3[Index::IRT], DIM3, _flux3);

    time2 = std::chrono::high_resolution_clock::now();
    timer["LR3->F3"] +=
        std::chrono::duration<double, std::milli>(time2 - time2e).count();
  }

  //// ------------ (5) Calculate flux divergence ------------ ////
  _div.set_(pcoord->forward(_flux1, _flux2, _flux3));

  auto time3 = std::chrono::high_resolution_clock::now();
  timer["F->D"] +=
      std::chrono::duration<double, std::milli>(time3 - time2).count();

  //// ------------ (6) Calculate external forcing ------------ ////
  auto du = -dt * _div;
  for (auto& f : forcings) f.forward(du, w, dt);

  auto time4 = std::chrono::high_resolution_clock::now();
  timer["W->R"] +=
      std::chrono::duration<double, std::milli>(time4 - time3).count();

  //// ------------ (7) Perform implicit correction ------------ ////
  torch::Tensor du0 = du.clone();
  // pvic->forward(w, du, gamma - 1., dt);
  torch::sub_out(_vic, du, du0);

  auto time5 = std::chrono::high_resolution_clock::now();
  timer["U->I"] +=
      std::chrono::duration<double, std::milli>(time5 - time4).count();

  return du;
}

void HydroImpl::fix_negative_dp_inplace(torch::Tensor wlr,
                                        torch::Tensor wdc) const {
  auto mask = torch::logical_or(wlr.select(1, Index::IDN) < 0.,
                                wlr.select(1, Index::IPR) < 0.);
  wlr.select(1, Index::IDN) =
      torch::where(mask, wdc.select(1, Index::IDN), wlr.select(1, Index::IDN));
  wlr.select(1, Index::IPR) =
      torch::where(mask, wdc.select(1, Index::IPR), wlr.select(1, Index::IPR));
}

void check_recon(torch::Tensor wlr, int nghost, int extend_x1, int extend_x2,
                 int extend_x3) {
  auto interior =
      get_interior(wlr.sizes(), nghost, extend_x1, extend_x2, extend_x3);

  int dim = extend_x1 == 1 ? 1 : (extend_x2 == 1 ? 2 : 3);
  TORCH_CHECK(
      wlr.index(interior).select(1, Index::IDN).min().item<double>() > 0.,
      "Negative density detected after reconstruction in dimension ", dim);
  TORCH_CHECK(
      wlr.index(interior).select(1, Index::IPR).min().item<double>() > 0.,
      "Negative pressure detected after reconstruction in dimension ", dim);
}

void check_eos(torch::Tensor w, int nghost) {
  auto interior = get_interior(w.sizes(), nghost);
  TORCH_CHECK(w.index(interior)[Index::IDN].min().item<double>() > 0.,
              "Negative density detected after EOS. ",
              "Suggestions: 1) Reducting the CFL number;",
              " 2) Activate EOS limiter and set the density floor");
  TORCH_CHECK(w.index(interior)[Index::IPR].min().item<double>() > 0.,
              "Negative pressure detected after EOS. ",
              "Suggestions: 1) Reducting the CFL number; ",
              " 2) Activate EOS limiter and set the pressure floor");
}

}  // namespace snap
