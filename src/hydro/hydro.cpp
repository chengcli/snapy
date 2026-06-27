#include <kintera/utils/format.hpp>

// C/C++
#include <algorithm>

// snap
#include <snap/snap.h>

#include <snap/mesh/meshblock.hpp>
#include <snap/utils/log.hpp>

#include "hydro.hpp"

namespace snap {
namespace {

bool fused_combo_supported(std::string const& eos_type,
                           std::string const& riemann_type) {
  return ((eos_type == "ideal-gas" || eos_type == "ideal-moist") &&
          (riemann_type == "lmars" || riemann_type == "hllc")) ||
         (eos_type == "shallow-water" && riemann_type == "shallow-roe");
}

bool fused_recon_type_supported(std::string const& type) {
  return type == "cp3" || type == "cp5" || type == "weno3" || type == "weno5";
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
    auto eos_type = options->eos()->type();
    auto riemann_type = options->riemann()->type();
    bool supported =
        u.is_cuda() && fused_combo_supported(eos_type, riemann_type) &&
        fused_recon_type_supported(precon1->pinterp1->options->type()) &&
        fused_recon_type_supported(precon23->pinterp1->options->type()) &&
        !other.count("solid") && !psed;
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
