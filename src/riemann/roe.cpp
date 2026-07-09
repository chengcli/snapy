// snap
#include <snap/snap.h>

#include <snap/hydro/hydro.hpp>

#include "../eos/ideal_moist.hpp"
#include "riemann_solver.hpp"

namespace snap {
void RoeSolverImpl::reset() {
  TORCH_CHECK(phydro, "[RoeSolver] parent is nullptr");
}

torch::Tensor RoeSolverImpl::forward(torch::Tensor wl, torch::Tensor wr,
                                     int dim, torch::Tensor flx,
                                     torch::Tensor face_pressure) {
  auto peos = phydro->peos;
  auto eos_type = peos->options->type();
  int nvar = wl.size(0);
  int ny = std::max<int>(nvar - ICY, 0);

  // dim, ivx, ivy, ivz
  // 3, IVX, IVY, iVZ
  // 2, IVX + 1, IVX + 2, IVX
  // 1, IVX + 2, IVX, IVX + 1
  auto ivx = IPR - dim;
  auto ivy = IVX + ((ivx - IVX) + 1) % 3;
  auto ivz = IVX + ((ivx - IVX) + 2) % 3;

  auto ul = peos->compute("W->U", {wl});
  auto ur = peos->compute("W->U", {wr});
  auto el = ul[IPR];
  auto gammal = peos->compute("W->A", {wl});
  auto cl = peos->compute("WA->L", {wl, gammal});

  auto er = ur[IPR];
  auto gammar = peos->compute("W->A", {wr});
  auto cr = peos->compute("WA->L", {wr, gammar});

  auto scalar_view = wl.sizes().vec();
  scalar_view[0] = 1;

  auto sqrtdl = torch::sqrt(wl[IDN]);
  auto sqrtdr = torch::sqrt(wr[IDN]);
  auto isdlpdr = 1.0 / (sqrtdl + sqrtdr);
  auto rhobar = sqrtdl * sqrtdr;

  //--- Step 2.  Compute Roe-averaged data from left- and right-states
  auto wroe = torch::zeros_like(wl);

  wroe[IDN] = sqrtdl * sqrtdr;
  wroe.narrow(0, IVX, 3) =
      (sqrtdl * wl.narrow(0, IVX, 3) + sqrtdr * wr.narrow(0, IVX, 3)) * isdlpdr;

  // Following Roe(1981), the enthalpy H=(E+P)/d is averaged for adiabatic
  // flows, rather than E or P directly.  sqrtdl*hl = sqrtdl*(el+pl)/dl =
  // (el+pl)/sqrtdl
  auto hl = (el + wl[IPR]) / wl[IDN];
  auto hr = (er + wr[IPR]) / wr[IDN];
  wroe[IPR] = (hl * sqrtdl + hr * sqrtdr) * isdlpdr;
  if (face_pressure.defined()) {
    face_pressure.copy_(wroe[IPR]);
  }

  //--- Step 3.  Compute L/R fluxes
  auto fl = torch::zeros_like(wl);
  auto fr = torch::zeros_like(wr);

  fl[IDN] = ul[IDN] * wl[ivx];
  fr[IDN] = ur[IDN] * wr[ivx];
  if (ny > 0) {
    fl.narrow(0, ICY, ny) = ul.narrow(0, ICY, ny) * wl[ivx].view(scalar_view);
    fr.narrow(0, ICY, ny) = ur.narrow(0, ICY, ny) * wr[ivx].view(scalar_view);
  }

  fl.narrow(0, IVX, 3) = wl[IDN] * wl[ivx] * wl.narrow(0, IVX, 3);
  fr.narrow(0, IVX, 3) = wr[IDN] * wr[ivx] * wr.narrow(0, IVX, 3);

  fl[ivx] += wl[IPR];
  fr[ivx] += wr[IPR];

  fl[IPR] = (el + wl[IPR]) * wl[ivx];
  fr[IPR] = (er + wr[IPR]) * wr[ivx];

  //--- Step 4.  Compute Roe fluxes.
  auto du = ur - ul;

  auto out = 0.5 * (fl + fr);

  auto vsq = wroe.narrow(0, IVX, 3).square().sum(0);
  auto gamma_roe = 0.5 * (gammal + gammar);
  auto offset = torch::zeros_like(wroe[IPR]);
  auto qbar_dry = (ul[IDN] / sqrtdl + ur[IDN] / sqrtdr) * isdlpdr;
  auto qbar = torch::Tensor();
  auto alpha_species = torch::Tensor();

  if (ny > 0) {
    auto sqrtdl_v = sqrtdl.view(scalar_view);
    auto sqrtdr_v = sqrtdr.view(scalar_view);
    auto isdlpdr_v = isdlpdr.view(scalar_view);
    qbar =
        (ul.narrow(0, ICY, ny) / sqrtdl_v + ur.narrow(0, ICY, ny) / sqrtdr_v) *
        isdlpdr_v;
  }

  if (eos_type == "ideal-moist") {
    auto moist = dynamic_cast<IdealMoistImpl*>(peos.get());
    TORCH_CHECK(moist != nullptr, "[RoeSolver] ideal-moist EOS cast failed");
    int nvapor = moist->pthermo->options->vapor_ids().size() - 1;
    if (ny > 0) {
      auto feps = torch::ones_like(wroe[IPR]);
      if (nvapor > 0) {
        feps += (qbar.narrow(0, 0, nvapor) *
                 moist->inv_mu_ratio_m1.narrow(0, 0, nvapor)
                     .to(qbar)
                     .view({nvapor, 1, 1, 1}))
                    .sum(0);
      }
      if (ny > nvapor) {
        feps -= qbar.narrow(0, nvapor, ny - nvapor).sum(0);
      }
      auto fsig =
          torch::ones_like(wroe[IPR]) +
          (qbar * moist->cv_ratio_m1.to(qbar).view({ny, 1, 1, 1})).sum(0);
      gamma_roe = 1.0 + (moist->options->gammad() - 1.0) * feps / fsig;
      offset = qbar_dry * moist->u0[0].to(wl);
      offset += (qbar * moist->u0.narrow(0, 1, ny).to(qbar).view({ny, 1, 1, 1}))
                    .sum(0);
    }
  }

  auto q = wroe[IPR] - 0.5 * vsq - offset;
  auto cs_sq = torch::clamp_min((gamma_roe - 1.0) * q, 1.0e-10);
  auto cs = torch::sqrt(cs_sq);
  auto lam_m = wroe[ivx] - cs;
  auto lam_0 = wroe[ivx];
  auto lam_p = wroe[ivx] + cs;
  auto spd_m = torch::abs(lam_m);
  auto spd_0 = torch::abs(lam_0);
  auto spd_p = torch::abs(lam_p);

  auto duv = wr[ivx] - wl[ivx];
  auto dv = wr[ivy] - wl[ivy];
  auto dw = wr[ivz] - wl[ivz];
  auto dp = wr[IPR] - wl[IPR];

  auto alpha_m = -0.5 * rhobar / cs * duv + 0.5 * dp / cs_sq;
  auto alpha_p = +0.5 * rhobar / cs * duv + 0.5 * dp / cs_sq;
  auto alpha_v = rhobar * dv;
  auto alpha_w = rhobar * dw;

  auto alpha_dry = ur[IDN] - ul[IDN] - dp / cs_sq * qbar_dry;
  auto alpha0 = alpha_dry.clone();
  auto llf_flag =
      torch::logical_or(ul[IDN] + alpha_m * qbar_dry < 0.0,
                        ul[IDN] + alpha_m * qbar_dry + alpha_dry < 0.0);

  out[IDN] -= 0.5 * (spd_m * alpha_m * qbar_dry + spd_0 * alpha_dry +
                     spd_p * alpha_p * qbar_dry);

  if (ny > 0) {
    auto sq_view = cs_sq.view(scalar_view);
    auto spd_m_v = spd_m.view(scalar_view);
    auto spd_0_v = spd_0.view(scalar_view);
    auto spd_p_v = spd_p.view(scalar_view);
    auto alpha_m_v = alpha_m.view(scalar_view);
    auto alpha_p_v = alpha_p.view(scalar_view);
    auto dp_v = dp.view(scalar_view);
    alpha_species =
        ur.narrow(0, ICY, ny) - ul.narrow(0, ICY, ny) - dp_v * qbar / sq_view;
    alpha0 += alpha_species.sum(0);

    out.narrow(0, ICY, ny) -=
        0.5 * (spd_m_v * alpha_m_v * qbar + spd_0_v * alpha_species +
               spd_p_v * alpha_p_v * qbar);

    auto species_first = ul.narrow(0, ICY, ny) + alpha_m_v * qbar;
    auto species_second = species_first + alpha_species;
    llf_flag = torch::logical_or(
        llf_flag, torch::logical_or(torch::any(species_first < 0.0, 0),
                                    torch::any(species_second < 0.0, 0)));
  }

  out[ivx] -=
      0.5 * (spd_m * alpha_m * (wroe[ivx] - cs) + spd_0 * (wroe[ivx] * alpha0) +
             spd_p * alpha_p * (wroe[ivx] + cs));
  out[ivy] -= 0.5 * (spd_m * alpha_m * wroe[ivy] +
                     spd_0 * (wroe[ivy] * alpha0 + alpha_v) +
                     spd_p * alpha_p * wroe[ivy]);
  out[ivz] -= 0.5 * (spd_m * alpha_m * wroe[ivz] +
                     spd_0 * (wroe[ivz] * alpha0 + alpha_w) +
                     spd_p * alpha_p * wroe[ivz]);
  out[IPR] -= 0.5 * (spd_m * alpha_m * (wroe[IPR] - wroe[ivx] * cs) +
                     spd_0 * (rhobar * (hr - hl - wroe[ivx] * duv) +
                              alpha0 * wroe[IPR] - dp) +
                     spd_p * alpha_p * (wroe[IPR] + wroe[ivx] * cs));

  //--- Step 5.  Overwrite with upwind flux if flow is supersonic
  auto evi = (lam_m > 0).to(out.scalar_type()).view(scalar_view);
  out = evi * fl + (1 - evi) * out;

  evi = (lam_p < 0).to(out.scalar_type()).view(scalar_view);
  out = evi * fr + (1 - evi) * out;

  //--- Step 6.  Overwrite with LLF flux if any of intermediate states are
  // negative
  auto cmax =
      0.5 * torch::max(torch::abs(wl[ivx]) + cl, torch::abs(wr[ivx]) + cr);
  auto llf_mask = llf_flag.to(out.scalar_type()).view(scalar_view);
  out = llf_mask * (0.5 * (fl + fr) - cmax.view(scalar_view) * du) +
        (1 - llf_mask) * out;

  flx.copy_(out);
  return flx;
}
}  // namespace snap
