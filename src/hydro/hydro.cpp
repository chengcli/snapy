#include <kintera/utils/format.hpp>

// C/C++
#include <algorithm>

// snap
#include <snap/snap.h>

#include <snap/mesh/meshblock.hpp>
#include <snap/utils/log.hpp>

#include "hydro.hpp"
#include "hydro_dispatch.hpp"

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

  _positivity_hits =
      register_buffer("positivity_hits", torch::zeros({1}, torch::kInt64));
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

torch::Tensor HydroImpl::_apply_implicit_correction(torch::Tensor& du,
                                                    torch::Tensor const& w,
                                                    double dt,
                                                    Variables const& other) {
  if (!picorr) return torch::Tensor();

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

  du[IPR].sub_(peos->internal_energy_offset(du));

  torch::Tensor gamma;
  if (options->eos()->type() == "aneos") {
    auto cs = peos->compute("W->L", {w});
    gamma = peos->compute("WL->A", {w, cs});
  } else {
    gamma = peos->compute("W->A", {wi});
  }
  auto correction = picorr->forward(du, wi, gamma, dt);
  du[IPR].add_(peos->internal_energy_offset(du));
  // picorr measured its delta after removing the EOS reference energy.
  // Diagnostics expose a conserved-state delta, so restore that reference
  // contribution using the corrected density and species tendencies.
  correction[IPR].add_(peos->internal_energy_offset(correction));

  return correction;
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

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
HydroImpl::_hydro_ref_x1(torch::Tensor const& w) const {
  auto pcoord = pmb->pcoord;
  int is = pcoord->il();
  int iu = pcoord->iu();
  double g = -options->grav()->grav1();  // downward magnitude (>0)

  // Global x1 reference across a vertical (nb1>1) decomposition: make the
  // hydrostatic reference continuous over the x1 process seams. The block
  // owning x1-outer anchors at the true domain top; every block below receives
  // the running seam-face pressure from the block above and passes on that
  // value plus its own interior hydrostatic drop (= its bottom-face pressure).
  // A serial top->bottom scan along the x1 process column. nb1 == 1 (no x1
  // neighbor) => an empty anchor tells the backend to compute the local top
  // anchor.
  torch::Tensor anchor;
  torch::Tensor
      kbot;  // [nc3,nc2,1] single global isentrope K = P_bot/rho_bot^gam
  torch::Tensor gam_global;
  int below = -1;
  int above = -1;
  bool x1_split = false;
  auto layout = pmb->get_layout();
  if (layout && layout->has_process_group() && !layout->options->periodic_z() &&
      layout->options->pz() >
          1) {  // pz==nb1: relay only across a SPLIT x1 column (else unmatched
                // send/recv at nb1=1)
    x1_split = true;
    auto iloc = layout->loc_of(layout->options->rank());
    above = layout->neighbor_rank(iloc, {0, 0, 1});   // toward x1-outer
    below = layout->neighbor_rank(iloc, {0, 0, -1});  // toward x1-inner
    constexpr int kWbRefTag = 0x7715;
    constexpr int kWbKbotTag = 0x7716;

    // Global (kbot, gamma) relay, bottom -> top. The density reference must be
    // ONE isentrope for the whole column: with a block-local kbot, adjacent
    // blocks decompose rho against different references and the two sides of
    // every x1 seam reconstruct different face states, making the dynamics
    // nb1-dependent. The block owning the physical bottom computes
    // (kbot, gamma) exactly as the nb1=1 path would; every block above
    // receives and forwards the same slabs.
    if (below >= 0) {
      // ONE tensor per message: ProcessGroupGloo::send rejects a multi-tensor
      // vector (ucx accepts it, which is why the ucx-only gate missed this).
      // Pack (kbot, gam) along the last axis, exactly as the seam-flux exchange
      // in hydro_forward.cpp packs its slab.
      std::vector<torch::Tensor> kbuf = {
          torch::empty({w.size(1), w.size(2), 2}, w.options())};
      layout->comm->recv(kbuf, below, kWbKbotTag)->wait();
      kbot = kbuf[0].narrow(-1, 0, 1).contiguous();
      gam_global = kbuf[0].narrow(-1, 1, 1).contiguous();
    } else {
      int is_loc = pcoord->il();
      gam_global =
          peos->compute("W->A", {w.narrow(-1, is_loc, 1)}).contiguous();
      kbot = (w[IPR].narrow(-1, is_loc, 1) /
              w[IDN].narrow(-1, is_loc, 1).pow(gam_global))
                 .contiguous();
    }
    if (above >= 0) {
      std::vector<torch::Tensor> sbuf = {
          torch::cat({kbot, gam_global}, -1).contiguous()};
      layout->comm->send(sbuf, above, kWbKbotTag)->wait();
    }

    if (above >= 0) {
      std::vector<torch::Tensor> rbuf = {
          torch::empty({w.size(1), w.size(2), 1}, w.options())};
      layout->comm->recv(rbuf, above, kWbRefTag)->wait();
      anchor = rbuf[0];
    }
  }

  if (x1_uniform_ < 0) {
    auto d = pcoord->dx1f.to(torch::kCPU);
    x1_uniform_ =
        ((d.max() - d.min()).item<double>() < 1e-10 * d.mean().item<double>())
            ? 1
            : 0;
  }

  auto gam = x1_split
                 ? gam_global
                 : peos->compute("W->A", {w.narrow(-1, is, 1)}).contiguous();
  auto ref_options = w.options();
  auto ref_sizes = w.sizes().slice(1).vec();
  auto psf_lo = torch::empty(ref_sizes, ref_options);
  auto psf_hi = torch::empty(ref_sizes, ref_options);
  auto pref = torch::empty(ref_sizes, ref_options);
  auto dsf = torch::empty(ref_sizes, ref_options);
  auto dref = torch::empty(ref_sizes, ref_options);
  bool phys_in = pmb->options->is_physical_boundary(0, 0, -1);
  bool phys_out = pmb->options->is_physical_boundary(0, 0, 1);

  auto dx1f = pcoord->dx1f.contiguous();
  at::native::call_hydro_ref_x1(w.device().type(), w, dx1f, anchor, gam, kbot,
                                psf_lo, psf_hi, pref, dsf, dref, is, iu, g,
                                x1_uniform_ == 1, phys_in, phys_out);

  if (below >= 0) {
    constexpr int kWbRefTag = 0x7715;
    std::vector<torch::Tensor> sbuf = {psf_lo.narrow(-1, is, 1).contiguous()};
    layout->comm->send(sbuf, below, kWbRefTag)->wait();
  }

  // Ghost-row reference exchange across x1 seams: overwrite this block's
  // ghost rows of (pref, dref) with the NEIGHBOR'S interior rows for the same
  // physical cells. The block-local ghost quadrature (w6e edge rows, and the
  // [lo,hi] guard whose accept/fallback decision is stencil-dependent) does
  // NOT reproduce the owner's interior values -- near a physical wall the
  // mismatch reaches ~1e2 Pa (measured), which enters
  // w' = w - ref as a spurious seam perturbation every step and makes the
  // dynamics nb1-dependent. After this exchange the perturbation field seen
  // by the reconstruction is identical on both sides of every seam, so the
  // seam-face states agree and the single-valued seam flux average becomes a
  // no-op. nb1=1: no seams, bit-unchanged.
  if (x1_split && (below >= 0 || above >= 0)) {
    constexpr int kWbGhostUpTag = 0x7717;
    constexpr int kWbGhostDnTag = 0x7718;
    int ng = is;  // il() == nghost
    std::vector<CommWorkPtr> sends;
    if (above >=
        0) {  // my top interior rows are the above block's lower ghosts
      std::vector<torch::Tensor> up = {
          torch::cat({pref.narrow(-1, iu - ng + 1, ng),
                      dref.narrow(-1, iu - ng + 1, ng)},
                     -1)
              .contiguous()};
      sends.push_back(layout->comm->send(up, above, kWbGhostUpTag));
    }
    if (below >=
        0) {  // my bottom interior rows are the below block's upper ghosts
      std::vector<torch::Tensor> dn = {
          torch::cat({pref.narrow(-1, is, ng), dref.narrow(-1, is, ng)}, -1)
              .contiguous()};
      sends.push_back(layout->comm->send(dn, below, kWbGhostDnTag));
    }
    if (below >= 0) {  // receive my lower ghost rows from below's top interior
      std::vector<torch::Tensor> rb = {
          torch::empty({w.size(1), w.size(2), 2 * ng}, w.options())};
      layout->comm->recv(rb, below, kWbGhostUpTag)->wait();
      pref.narrow(-1, 0, ng).copy_(rb[0].narrow(-1, 0, ng));
      dref.narrow(-1, 0, ng).copy_(rb[0].narrow(-1, ng, ng));
    }
    if (above >=
        0) {  // receive my upper ghost rows from above's bottom interior
      std::vector<torch::Tensor> ra = {
          torch::empty({w.size(1), w.size(2), 2 * ng}, w.options())};
      layout->comm->recv(ra, above, kWbGhostDnTag)->wait();
      pref.narrow(-1, iu + 1, ng).copy_(ra[0].narrow(-1, 0, ng));
      dref.narrow(-1, iu + 1, ng).copy_(ra[0].narrow(-1, ng, ng));
    }
    for (auto& sw : sends) sw->wait();
  }

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
