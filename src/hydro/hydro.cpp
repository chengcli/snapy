#include <kintera/utils/format.hpp>

// C/C++
#include <algorithm>
#include <limits>

// snap
#include <snap/snap.h>

#include <snap/mesh/meshblock.hpp>
#include <snap/utils/log.hpp>

#include "hydro.hpp"

namespace snap {
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
    _face_pressure1 =
        register_buffer("P1", torch::zeros({nc3, nc2, nc1}, torch::kFloat64));
  } else {
    _flux1 = register_buffer("F1", torch::Tensor());
    _face_pressure1 = register_buffer("P1", torch::Tensor());
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

torch::Tensor HydroImpl::implicit_mass_correction() const {
  return picorr ? picorr->mass_correction() : torch::Tensor();
}

void HydroImpl::_apply_implicit_correction(torch::Tensor& du,
                                           torch::Tensor const& w, double dt,
                                           Variables const& other) {
  if (!picorr) return;

  // Implicit x1 solve has no cross-rank coupling, so nb1 > 1 would silently
  // solve each rank's own sub-column. Full column <=> both x1 faces physical.
  TORCH_CHECK(pmb->options->is_physical_boundary(0, 0, -1) &&
                  pmb->options->is_physical_boundary(0, 0, 1),
              "[Hydro] implicit scheme requires nb1 = 1 (no x1 decomposition): "
              "the vertical solve has no cross-rank coupling.");

  torch::Tensor wi;
  if (other.count("solid")) {
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
}

void HydroImpl::_revise_x1inner_lr(torch::Tensor const& wl,
                                   torch::Tensor const& wr) {
  int is = pmb->pcoord->il();
  wl[IPR].narrow(-1, is, 1) = wr[IPR].narrow(-1, is, 1);
  wl[IDN].narrow(-1, is, 1) = wr[IDN].narrow(-1, is, 1);
}

void HydroImpl::_revise_x1outer_lr(torch::Tensor const& wl,
                                   torch::Tensor const& wr) {
  int ie = pmb->pcoord->iu();
  wr[IPR].narrow(-1, ie + 1, 1) = wl[IPR].narrow(-1, ie + 1, 1);
  wr[IDN].narrow(-1, ie + 1, 1) = wl[IDN].narrow(-1, ie + 1, 1);
}

void HydroImpl::_revise_x1inner_ghost(torch::Tensor const& w) {
  auto pcoord = pmb->pcoord;
  int is = pcoord->il();
  auto grav = -options->grav()->grav1();

  auto gamma = peos->compute("W->A", {w.narrow(-1, is, 1)});
  auto gm = gamma - 1.;
  auto a = gm / gamma;
  auto K = w[IPR].narrow(-1, is, 1) / w[IDN].narrow(-1, is, 1).pow(gamma);

  for (int n = 0; n < pcoord->options->nghost(); ++n) {
    auto dz = pmb->pcoord->dx1v[is - n - 1];
    auto h =
        w[IPR].narrow(-1, is - n, 1).pow(a) + a * grav * dz / K.pow(1. / gamma);
    w[IPR].narrow(-1, is - n - 1, 1) = h.pow(1. / a);
    w[IDN].narrow(-1, is - n - 1, 1) =
        (w[IPR].narrow(-1, is - n - 1, 1) / K).pow(1. / gamma);
  }
}

void HydroImpl::_revise_x1outer_ghost(torch::Tensor const& w) {
  auto pcoord = pmb->pcoord;
  int ie = pcoord->iu();
  auto grav = -options->grav()->grav1();

  auto gamma = peos->compute("W->A", {w.narrow(-1, ie, 1)});
  auto gm = gamma - 1.;
  auto a = gm / gamma;
  auto K = w[IPR].narrow(-1, ie, 1) / w[IDN].narrow(-1, ie, 1).pow(gamma);

  for (int n = 0; n < pcoord->options->nghost(); ++n) {
    auto dz = pmb->pcoord->dx1v[ie + n];
    auto h =
        w[IPR].narrow(-1, ie + n, 1).pow(a) - a * grav * dz / K.pow(1. / gamma);
    w[IPR].narrow(-1, ie + n + 1, 1) = h.pow(1. / a);
    w[IDN].narrow(-1, ie + n + 1, 1) =
        (w[IPR].narrow(-1, ie + n + 1, 1) / K).pow(1. / gamma);
  }
}

std::vector<torch::Tensor> HydroImpl::_hydro_ref_x1(
    torch::Tensor const& w) const {
  auto pcoord = pmb->pcoord;
  int is = pcoord->il();
  int iu = pcoord->iu();
  int nc1 = w.size(-1);
  double g = -options->grav()->grav1();  // downward magnitude (>0)

  auto rho = w[IDN];                 // [nc3,nc2,nc1]
  auto dx1f = pcoord->dx1f;          // [nc1]
  auto dp = g * rho * dx1f;          // hydrostatic drop across each cell
  auto cum = torch::cumsum(dp, -1);  // cum(i) = sum_{0..i} dp

  // Face pressures integrated DOWNWARD from an isothermal half-cell anchor
  // above the top interior cell. The top anchor keeps the perturbation small
  // relative to P everywhere (a bottom anchor piles the integration residual
  // into the low-pressure top, where it exceeds P itself). The anchor value
  // cancels from the momentum balance for any cell estimator that is LINEAR
  // in the face values.
  auto RdT_top = w[IPR].select(-1, iu) / rho.select(-1, iu);  // [nc3,nc2]
  auto anchor =
      (w[IPR].select(-1, iu) * torch::exp(-g * 0.5 * dx1f[iu] / RdT_top))
          .unsqueeze(-1);                          // [nc3,nc2,1]
  auto cum_iu = cum.select(-1, iu).unsqueeze(-1);  // [nc3,nc2,1]
  auto psf_lo = anchor + cum_iu - cum + dp;  // pressure at lower face of cell i
  auto psf_hi = psf_lo - dp;                 // pressure at upper face of cell i

  // In an extreme domain the hydrostatic continuation through the outer
  // ghost cells could reach <= 0 and poison the near-boundary stencils with
  // NaN via the pow below. Clamping is the identity for any healthy column.
  psf_lo.clamp_min_(std::numeric_limits<double>::min());
  psf_hi.clamp_min_(std::numeric_limits<double>::min());

  // Cell pressure reference = estimate of the cell AVERAGE of the exact
  // hydrostatic pressure from the face values. On a uniform grid, a six-face
  // quadrature (exact through degree 5) guarded to fall back to the 2-point
  // mean where the wide stencil leaves the cell's own face interval (kinks).
  // The 2-point log-mean is exact only for an isothermal layer -- O(dz^2) on
  // any other stratification, which leaves e.g. a dry adiabat with p' != 0
  // and a phantom force. Non-uniform grid: the quadrature weights do not
  // apply; keep the log-mean.
  if (x1_uniform_ < 0) {
    auto d = dx1f.to(torch::kCPU);
    x1_uniform_ =
        ((d.max() - d.min()).item<double>() < 1e-10 * d.mean().item<double>())
            ? 1
            : 0;
  }

  torch::Tensor pref;
  if (x1_uniform_ == 1) {
    pref = 0.5 * (psf_lo + psf_hi);
    // face pressures 0..nc1: face i = lower face of cell i, plus the top face
    auto faces = torch::cat({psf_lo, psf_hi.narrow(-1, nc1 - 1, 1)}, -1);
    constexpr double w6[6] = {11. / 1440., -31. / 480., 401. / 720.,
                              401. / 720., -31. / 480., 11. / 1440.};
    auto six = w6[0] * faces.narrow(-1, 0, nc1 - 4);
    for (int k = 1; k < 6; ++k) {
      six = six + w6[k] * faces.narrow(-1, k, nc1 - 4);
    }
    auto lo = torch::minimum(psf_lo, psf_hi).narrow(-1, 2, nc1 - 4);
    auto hi = torch::maximum(psf_lo, psf_hi).narrow(-1, 2, nc1 - 4);
    auto mid = pref.narrow(-1, 2, nc1 - 4);
    mid.copy_(torch::where((six >= lo) & (six <= hi), six, mid));
  } else {
    auto ratio = psf_lo / psf_hi;
    pref = torch::where((ratio - 1.).abs() < 1e-6, 0.5 * (psf_lo + psf_hi),
                        dp / torch::log(ratio));
  }

  // Isentropic density reference from the bottom cell:
  // rho_ref(P)=(P/K)^(1/gamma), K = P_bot / rho_bot^gamma. Decomposing density
  // too keeps the reconstructed face (rho, P) pair thermodynamically consistent
  // -- reconstructing full density against a perturbation pressure was the
  // destabilizing mismatch.
  auto gam = peos->compute("W->A", {w.narrow(-1, is, 1)});  // [nc3,nc2,1]
  auto Kbot = w[IPR].select(-1, is).unsqueeze(-1) /
              rho.select(-1, is).unsqueeze(-1).pow(gam);  // [nc3,nc2,1]
  auto dref = (pref / Kbot).pow(1.0 / gam);
  auto dsf = (psf_lo / Kbot).pow(1.0 / gam);

  return {psf_lo, pref, dsf, dref};
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
