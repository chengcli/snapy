// snap
#include "hydro.hpp"

#include <snap/snap.h>

#include <snap/registry.hpp>

namespace snap {

HydroImpl::HydroImpl(const HydroOptions& options_) : options(options_) {
  reset();
}

void HydroImpl::reset() {
  //// ---- (1) set up coordinate model ---- ////
  pcoord = register_module_op(this, "coord", options.coord());
  options.coord() = pcoord->options;

  //// ---- (2) set up equation-of-state model ---- ////
  peos = register_module_op(this, "eos", options.eos());
  options.eos() = peos->options;

  //// ---- (3) set up primitive projector model ---- ////
  pproj = register_module("proj", PrimitiveProjector(options.proj()));
  options.proj() = pproj->options;

  //// ---- (4) set up reconstruction-x1 model ---- ////
  precon1 = register_module("recon1", Reconstruct(options.recon1()));
  options.recon1() = precon1->options;

  //// ---- (5) set up reconstruction-x23 model ---- ////
  precon23 = register_module("recon23", Reconstruct(options.recon23()));
  options.recon23() = precon23->options;

  //// ---- (6) set up riemann-solver model ---- ////
  priemann = register_module_op(this, "riemann", options.riemann());
  options.riemann() = priemann->options;

  //// ---- (7) set up internal boundary ---- ////
  pib = register_module("ib", InternalBoundary(options.ib()));
  options.ib() = pib->options;

  //// ---- (8) set up implicit solver ---- ////
  pimp = register_module("imp", ImplicitCorrection(options.imp()));
  options.imp() = pimp->options;

  //// ---- (9) set up sedimentation ---- ////
  psed = register_module("sed", SedHydro(options.sed()));
  options.sed() = psed->options;

  //// ---- (10) set up forcings ---- ////
  std::vector<std::string> forcing_names;
  if (options.grav().grav1() != 0.0 || options.grav().grav2() != 0.0 ||
      options.grav().grav3() != 0.0) {
    if (!options.disable_dynamics()) {
      forcings.push_back(torch::nn::AnyModule(ConstGravity(options.grav())));
      forcing_names.push_back("const-gravity");
    }
  }

  if (options.coriolis().omega1() != 0.0 ||
      options.coriolis().omega2() != 0.0 ||
      options.coriolis().omega3() != 0.0) {
    forcings.push_back(torch::nn::AnyModule(Coriolis123(options.coriolis())));
    forcing_names.push_back("coriolis");
  }

  if (options.coriolis().omegax() != 0.0 ||
      options.coriolis().omegay() != 0.0 ||
      options.coriolis().omegaz() != 0.0) {
    forcings.push_back(torch::nn::AnyModule(CoriolisXYZ(options.coriolis())));
    if (std::find(forcing_names.begin(), forcing_names.end(), "coriolis") !=
        forcing_names.end()) {
      TORCH_CHECK(false,
                  "CoriolisXYZ cannot be used together with Coriolis123. "
                  "Please choose one of them.");
    }
    forcing_names.push_back("coriolis");
  }

  if (options.fricHeat().grav() != 0.0) {
    forcings.push_back(torch::nn::AnyModule(FricHeat(options.fricHeat())));
    forcing_names.push_back("fric-heat");
  }

  if (options.bodyHeat().dTdt() != 0.0) {
    forcings.push_back(torch::nn::AnyModule(BodyHeat(options.bodyHeat())));
    forcing_names.push_back("body-heat");
  }

  if (options.topCool().flux() != 0.0) {
    forcings.push_back(torch::nn::AnyModule(TopCool(options.topCool())));
    forcing_names.push_back("top-cool");
  }

  if (options.botHeat().flux() != 0.0) {
    forcings.push_back(torch::nn::AnyModule(BotHeat(options.botHeat())));
    forcing_names.push_back("bot-heat");
  }

  if (options.relaxBotComp().tau() != 0.0) {
    forcings.push_back(
        torch::nn::AnyModule(RelaxBotComp(options.relaxBotComp())));
    forcing_names.push_back("relax-bot-comp");
  }

  if (options.relaxBotTemp().tau() != 0.0) {
    forcings.push_back(
        torch::nn::AnyModule(RelaxBotTemp(options.relaxBotTemp())));
    forcing_names.push_back("relax-bot-temp");
  }

  if (options.relaxBotVelo().tau() != 0.0) {
    forcings.push_back(
        torch::nn::AnyModule(RelaxBotVelo(options.relaxBotVelo())));
    forcing_names.push_back("relax-bot-velo");
  }

  if (options.topSpongeLyr().tau() != 0.0 &&
      options.topSpongeLyr().width() > 0.0) {
    forcings.push_back(
        torch::nn::AnyModule(TopSpongeLyr(options.topSpongeLyr())));
    forcing_names.push_back("top-sponge-lyr");
  }

  if (options.botSpongeLyr().tau() != 0.0 &&
      options.botSpongeLyr().width() > 0.0) {
    forcings.push_back(
        torch::nn::AnyModule(BotSpongeLyr(options.botSpongeLyr())));
    forcing_names.push_back("bot-sponge-lyr");
  }

  //// ---- (11) register all forcings ---- ////
  for (auto i = 0; i < forcings.size(); i++) {
    register_module(forcing_names[i], forcings[i].ptr());
  }

  //// ---- (12) populate buffers ---- ////
  int nc1 = options.coord().nc1();
  int nc2 = options.coord().nc2();
  int nc3 = options.coord().nc3();
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
  // should be preceeded by initialize, W->I, or W->U
  torch::Tensor cs;
  if (options.eos().type() == "aneos") {
    cs = peos->compute("W->L", {w});
  } else {
    auto gamma = peos->compute("W->A", {w});
    cs = peos->compute("WA->L", {w, gamma});
  }

  if (solid.defined()) {
    cs = torch::where(solid, 1.e-8, cs);
  }

  double dt1 = 1.e9, dt2 = 1.e9, dt3 = 1.e9;

  if ((cs.size(2) > 1) &&
      (!(pimp->options.scheme() & 1) || (cs.size(0) == 1 && cs.size(1) == 1))) {
    dt1 = torch::min(pcoord->center_width1() / (w[IVX].abs() + cs))
              .item<double>();
  }

  if ((cs.size(1) > 1) && (!((pimp->options.scheme() >> 1) & 1))) {
    dt2 = torch::min(pcoord->center_width2() / (w[IVY].abs() + cs))
              .item<double>();
  }

  if ((cs.size(0) > 1) && (!((pimp->options.scheme() >> 2) & 1))) {
    dt3 = torch::min(pcoord->center_width3() / (w[IVZ].abs() + cs))
              .item<double>();
  }

  return std::min({dt1, dt2, dt3});
}

torch::Tensor HydroImpl::forward(double dt, torch::Tensor u,
                                 Variables const& other) {
  enum { DIM1 = 3, DIM2 = 2, DIM3 = 1 };
  bool has_solid = other.count("solid");

  //// ------------ (1) Calculate Primitives ------------ ////
  auto const& w = other.at("hydro_w");

  peos->forward(u, w);
  if (has_solid) {
    pib->mark_prim_solid_(w, other.at("solid"));
  }

  auto temp = peos->compute("W->T", {w});

  //// ------------ (2) Calculate dimension 1 flux ------------ ////
  std::chrono::high_resolution_clock::time_point time2;

  if (u.size(DIM1) > 1) {
    auto wp = pproj->forward(w, pcoord->dx1f);
    auto wtmp = precon1->forward(wp, DIM1);

    pproj->restore_inplace(wtmp);
    auto wlr1 = has_solid ? pib->forward(wtmp, DIM1, other.at("solid")) : wtmp;

    if (!options.disable_dynamics()) {
      priemann->forward(wlr1[ILT], wlr1[IRT], DIM1, _flux1);
    }

    // add sedimentation flux
    psed->forward(w, _flux1);
  }

  //// ------------ (3) Calculate dimension 2 flux ------------ ////
  if (u.size(DIM2) > 1) {
    auto wtmp = precon23->forward(w, DIM2);
    auto wlr2 = has_solid ? pib->forward(wtmp, DIM2, other.at("solid")) : wtmp;
    if (!options.disable_dynamics()) {
      priemann->forward(wlr2[ILT], wlr2[IRT], DIM2, _flux2);
    }
  }

  //// ------------ (4) Calculate dimension 3 flux ------------ ////
  if (u.size(DIM3) > 1) {
    auto wtmp = precon23->forward(w, DIM3);

    auto wlr3 = has_solid ? pib->forward(wtmp, DIM3, other.at("solid")) : wtmp;
    if (!options.disable_dynamics()) {
      priemann->forward(wlr3[ILT], wlr3[IRT], DIM3, _flux3);
    }
  }

  //// ------------ (5) Calculate flux divergence ------------ ////
  _div.set_(pcoord->forward(w, _flux1, _flux2, _flux3));

  //// ------------ (6) Calculate external forcing ------------ ////
  auto du = -dt * _div;
  for (auto& f : forcings) f.forward(du, w, temp, dt);

  //// ------------ (7) Perform implicit correction ------------ ////
  torch::Tensor wi;
  if (has_solid) {
    wi = torch::where(other.at("solid").unsqueeze(0).expand_as(w),
                      other.at("fill_solid_hydro_w"), w);
    du.masked_fill_(other.at("solid").unsqueeze(0).expand_as(du), 0.0);
  } else {
    wi = w;
  }

  torch::Tensor gamma;
  if (options.eos().type() == "aneos") {
    auto cs = peos->compute("W->L", {w});
    gamma = peos->compute("WL->A", {w, cs});
  } else {
    gamma = peos->compute("W->A", {wi});
  }
  _imp.set_(pimp->forward(du, wi, gamma, dt));

  return du;
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
