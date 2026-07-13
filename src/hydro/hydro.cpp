#include <kintera/utils/format.hpp>

// C/C++
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <string>

// snap
#include <snap/snap.h>

#include <snap/mesh/meshblock.hpp>
#include <snap/utils/log.hpp>

#include "hydro.hpp"

namespace snap {
namespace {

bool fused_runtime_supported(torch::Tensor const& u, Variables const& other) {
  return u.is_cuda() && !other.count("solid");
}

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

torch::Tensor HydroImpl::forward(double dt, torch::Tensor u,
                                 Variables const& other) {
  if (options->fused_recon_riemann()) {
    bool supported = fused_runtime_supported(u, other);
    // CPU fused path (fused_recon_riemann_cpu.cpp): cartesian coordinates
    // only -- the cubed-sphere metric/seam exchange is CUDA-only -- no
    // sedimentation or solid boundaries, and not the roe solver (the CPU
    // kernel implements lmars/hllc/shallow-roe only). Under FUSED=auto only
    // the ideal-gas EOS takes the CPU fused path (the combination validated
    // against the staged path); ideal-moist / shallow-water on CPU require an
    // explicit FUSED=on. CUDA tensors take the exact same decision as before.
    if (!supported && u.device().is_cpu() && !other.count("solid") && !psed &&
        options->riemann()->type() != "roe" &&
        pmb->pcoord->options->type() == "cartesian" &&
        pmb->get_layout()->options->type() != "cubed-sphere") {
      char const* fused_env = std::getenv("FUSED");
      bool explicit_on = fused_env != nullptr &&
                         (std::string(fused_env) == "on" ||
                          std::string(fused_env) == "ON" ||
                          std::string(fused_env) == "On" ||
                          std::string(fused_env) == "1");
      supported = explicit_on || options->eos()->type() == "ideal-gas";
    }
    if (!supported) {
      return _forward_staged(dt, u, other);
    }
    return _forward_fused(dt, u, other);
  }
  return _forward_staged(dt, u, other);
}

torch::Tensor HydroImpl::implicit_mass_correction() const {
  return picorr ? picorr->mass_correction() : torch::Tensor();
}

void HydroImpl::_apply_implicit_correction(torch::Tensor& du,
                                           torch::Tensor const& w, double dt,
                                           Variables const& other) {
  if (!picorr) return;
  // scheme 0 is the null operation: on CPU skip the wrapper too (du.clone,
  // energy-offset round trip, W->A temporary) -- ~0.1 ms/step of dispatch.
  // Gated to CPU so GPU behavior is exactly unchanged.
  if (picorr->options->scheme() == 0 && du.device().is_cpu()) return;

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

// scalar CPU implementation of the hydrostatic ghost-cell extrapolation for
// constant-gamma (ideal-gas) EOS; mirrors the tensor code in
// _revise_x1{inner,outer}_ghost term for term
void HydroImpl::_revise_x1_ghost_cpu_ideal(torch::Tensor const& w,
                                           bool inner) {
  auto pcoord = pmb->pcoord;
  int nghost = pcoord->options->nghost();
  int anchor = inner ? pcoord->il() : pcoord->iu();
  double grav = -options->grav()->grav1();
  double gamma = options->eos()->gammad();
  double gm = gamma - 1.;
  double a = gm / gamma;
  double inv_gamma = 1. / gamma;
  double inv_a = 1. / a;
  double ag = a * grav;

  int64_t nc3 = w.size(1), nc2 = w.size(2), nc1 = w.size(3);
  double* base = w.data_ptr<double>();
  double* rho = base + IDN * nc3 * nc2 * nc1;
  double* prs = base + IPR * nc3 * nc2 * nc1;
  double const* dx1v = pcoord->dx1v.data_ptr<double>();

  for (int64_t k = 0; k < nc3; ++k) {
    for (int64_t j = 0; j < nc2; ++j) {
      int64_t off = (k * nc2 + j) * nc1;
      double K = prs[off + anchor] / std::pow(rho[off + anchor], gamma);
      double Kpow = std::pow(K, inv_gamma);
      for (int n = 0; n < nghost; ++n) {
        int src = inner ? anchor - n : anchor + n;
        int dst = inner ? src - 1 : src + 1;
        double dz = dx1v[inner ? dst : src];
        double h = std::pow(prs[off + src], a) +
                   (inner ? ag * dz / Kpow : -(ag * dz / Kpow));
        prs[off + dst] = std::pow(h, inv_a);
        rho[off + dst] = std::pow(prs[off + dst] / K, inv_gamma);
      }
    }
  }
}

void HydroImpl::_revise_x1inner_ghost(torch::Tensor const& w) {
  auto pcoord = pmb->pcoord;
  int is = pcoord->il();
  auto grav = -options->grav()->grav1();

  // CPU fast path: ideal-gas gamma is a scalar and the recurrence touches
  // only nghost cells per column; a direct loop replaces ~40 dispatched
  // tensor ops per call (measured 0.69 ms/step on a 64x4 block). Same
  // formulas and evaluation order as the tensor code below; falls through
  // for CUDA tensors and non-ideal-gas EOS.
  if (w.device().is_cpu() && w.is_contiguous() &&
      w.scalar_type() == torch::kFloat64 &&
      peos->options->type() == "ideal-gas") {
    _revise_x1_ghost_cpu_ideal(w, /*inner=*/true);
    return;
  }

  auto gamma = peos->compute("W->A", {w.narrow(-1, is, 1)});
  auto gm = gamma - 1.;
  auto a = gm / gamma;
  auto K = w[IPR].narrow(-1, is, 1) / w[IDN].narrow(-1, is, 1).pow(gamma);

  // loop invariants (were recomputed every iteration; same values, same bits)
  auto inv_gamma = 1. / gamma;
  auto inv_a = 1. / a;
  auto ag = a * grav;
  auto Kpow = K.pow(inv_gamma);

  for (int n = 0; n < pcoord->options->nghost(); ++n) {
    auto dz = pmb->pcoord->dx1v[is - n - 1];
    auto h = w[IPR].narrow(-1, is - n, 1).pow(a) + ag * dz / Kpow;
    w[IPR].narrow(-1, is - n - 1, 1) = h.pow(inv_a);
    w[IDN].narrow(-1, is - n - 1, 1) =
        (w[IPR].narrow(-1, is - n - 1, 1) / K).pow(inv_gamma);
  }
}

void HydroImpl::_revise_x1outer_ghost(torch::Tensor const& w) {
  auto pcoord = pmb->pcoord;
  int ie = pcoord->iu();
  auto grav = -options->grav()->grav1();

  if (w.device().is_cpu() && w.is_contiguous() &&
      w.scalar_type() == torch::kFloat64 &&
      peos->options->type() == "ideal-gas") {
    _revise_x1_ghost_cpu_ideal(w, /*inner=*/false);
    return;
  }

  auto gamma = peos->compute("W->A", {w.narrow(-1, ie, 1)});
  auto gm = gamma - 1.;
  auto a = gm / gamma;
  auto K = w[IPR].narrow(-1, ie, 1) / w[IDN].narrow(-1, ie, 1).pow(gamma);

  // loop invariants (were recomputed every iteration; same values, same bits)
  auto inv_gamma = 1. / gamma;
  auto inv_a = 1. / a;
  auto ag = a * grav;
  auto Kpow = K.pow(inv_gamma);

  for (int n = 0; n < pcoord->options->nghost(); ++n) {
    auto dz = pmb->pcoord->dx1v[ie + n];
    auto h = w[IPR].narrow(-1, ie + n, 1).pow(a) - ag * dz / Kpow;
    w[IPR].narrow(-1, ie + n + 1, 1) = h.pow(inv_a);
    w[IDN].narrow(-1, ie + n + 1, 1) =
        (w[IPR].narrow(-1, ie + n + 1, 1) / K).pow(inv_gamma);
  }
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
