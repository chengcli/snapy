// snap
#include <snap/snap.h>

#include <snap/registry.hpp>

#include "riemann_solver.hpp"

namespace snap {
torch::Tensor _compute_uroe(torch::Tensor wroe, EquationOfState const& peos) {
  return wroe;
}

void RoeSolverImpl::reset() {
  // set up equation-of-state model
  peos = register_module_op(this, "eos", options.eos());
}

torch::Tensor RoeSolverImpl::forward(torch::Tensor wl, torch::Tensor wr,
                                     int dim, torch::Tensor flx) {
  using Index::IDN;
  using Index::IPR;
  using Index::IVX;

  // dim, ivx, ivy, ivz
  // 3, IVX, IVY, iVZ
  // 2, IVX + 1, IVX + 2, IVX
  // 1, IVX + 2, IVX, IVX + 1
  auto ivx = IPR - dim;
  auto ivy = IVX + ((ivx - IVX) + 1) % 3;
  auto ivz = IVX + ((ivx - IVX) + 2) % 3;

  auto el = peos->compute("W->U", {wl});
  auto gammal = peos->compute("W->A", {wl});
  auto cl = peos->compute("WA->L", {wl, gammal});

  auto er = peos->compute("W->U", {wr});
  auto gammar = peos->compute("W->A", {wr});
  auto cr = peos->compute("WA->L", {wr, gammar});

  //--- Step 2.  Compute Roe-averaged data from left- and right-states
  auto wroe = torch::zeros_like(wl);

  auto sqrtdl = torch::sqrt(wl[IDN]);
  auto sqrtdr = torch::sqrt(wr[IDN]);
  auto isdlpdr = 1.0 / (sqrtdl + sqrtdr);

  wroe[IDN] = sqrtdl * sqrtdr;
  wroe.narrow(0, IVX, 3) =
      (sqrtdl * wl.narrow(0, IVX, 3) + sqrtdr * wr.narrow(0, IVX, 3)) * isdlpdr;

  // Following Roe(1981), the enthalpy H=(E+P)/d is averaged for adiabatic
  // flows, rather than E or P directly.  sqrtdl*hl = sqrtdl*(el+pl)/dl =
  // (el+pl)/sqrtdl
  wroe[IPR] = ((el + wl[IPR]) / sqrtdl + (er + wr[IPR]) / sqrtdr) * isdlpdr;

  //--- Step 3.  Compute L/R fluxes
  auto fl = torch::zeros_like(wl);
  auto fr = torch::zeros_like(wr);

  fl[IDN] = wl[IDN] * wl[ivx];
  fr[IDN] = wr[IDN] * wr[ivx];

  fl.narrow(0, IVX, 3) = fl[IDN] * wl.narrow(0, IVX, 3);
  fr.narrow(0, IVX, 3) = fr[IDN] * wr.narrow(0, IVX, 3);

  fl[ivx] += wl[IPR];
  fr[ivx] += wr[IPR];

  fl[IPR] = (el + wl[IPR]) * wl[ivx];
  fr[IPR] = (er + wr[IPR]) * wr[ivx];

  //--- Step 4.  Compute Roe fluxes.
  auto du = torch::zeros_like(wroe);

  du[IDN] = wr[IDN] - wl[IDN];
  du.narrow(0, IVX, 3) =
      wr[IDN] * wr.narrow(0, IVX, 3) - wl[IDN] * wl.narrow(0, IVX, 3);
  du[IPR] = er - el;

  flx.set_(0.5 * (fl + fr));

  auto vsq = wroe.narrow(0, IVX, 3).square().sum(0);
  auto q = wroe[IPR] - 0.5 * vsq;
  auto qi = (q < 0.).to(torch::kInt);

  // FIXME: compute uroe
  auto uroe = _compute_uroe(wroe, peos);

  auto wroe1 = peos->compute("U->W", {uroe});
  auto gamma_roe = peos->compute("W->A", {wroe1});

  auto cs = peos->compute("WA->L", {wroe1, gamma_roe});
  auto cs_sq = cs.square();
  auto gm1_roe = gamma_roe - 1.0;

  // Compute eigenvalues (eq. B2)
  auto ev = torch::zeros_like(du);
  ev[0] = wroe[ivx] - cs;
  ev[1] = wroe[ivx];
  ev[2] = wroe[ivx];
  ev[3] = wroe[ivx];
  ev[4] = wroe[ivx] + cs;

  // Compute projection of dU onto L-eigenvectors using matrix elements from
  // eq. B4
  auto a = torch::zeros_like(du);
  auto na = 0.5 / cs_sq;
  a[0] = (du[0] * (0.5 * gm1_roe * vsq + wroe[ivx] * cs) -
          du[ivx] * (gm1_roe * wroe[ivx] + cs) - du[ivy] * gm1_roe * wroe[ivy] -
          du[ivz] * gm1_roe * wroe[ivz] + du[4] * gm1_roe) *
         na;

  a[1] = -du[0] * wroe[ivy] + du[ivy];
  a[2] = -du[0] * wroe[ivz] + du[ivz];

  auto qa = gm1_roe / cs_sq;
  a[3] = du[0] * (1.0 - na * gm1_roe * vsq) + du[ivx] * qa * wroe[ivx] +
         du[ivy] * qa * wroe[ivy] + du[ivz] * qa * wroe[ivz] - du[4] * qa;

  a[4] = (du[0] * (0.5 * gm1_roe * vsq - wroe[ivx] * cs) -
          du[ivx] * (gm1_roe * wroe[ivx] - cs) - du[ivy] * gm1_roe * wroe[ivy] -
          du[ivz] * gm1_roe * wroe[ivz] + du[4] * gm1_roe) *
         na;

  auto coeff = -0.5 * torch::abs(ev) * a;

  // compute density in intermediate states and check that it is positive,
  // set flag This requires computing the [0][*] components of the
  // right-eigenmatrix
  auto llf_flag =
      (torch::logical_or(wl[IDN] + a[0] < 0.0, wl[IDN] + a[0] + a[3] < 0))
          .to(torch::kInt);

  // Now multiply projection with R-eigenvectors from eq. B3 and SUM into
  // output fluxes
  flx[IDN] += coeff[0] + coeff[3] + coeff[4];

  flx[ivx] += coeff[0] * (wroe[ivx] - cs) + coeff[3] * wroe[ivx] +
              coeff[4] * (wroe[ivx] + cs);

  flx[ivy] += coeff[0] * wroe[ivy] + coeff[1] + coeff[3] * wroe[ivy] +
              coeff[4] * wroe[ivy];

  flx[ivz] += coeff[0] * wroe[ivz] + coeff[2] + coeff[3] * wroe[ivz] +
              coeff[4] * wroe[ivz];

  flx[IPR] += coeff[0] * (wroe[IPR] - wroe[ivx] * cs) + coeff[1] * wroe[ivy] +
              coeff[2] * wroe[ivz] + coeff[3] * 0.5 * vsq +
              coeff[4] * (wroe[IPR] + wroe[ivx] * cs);

  //--- Step 5.  Overwrite with upwind flux if flow is supersonic
  auto evi = (ev[0] > 0).to(torch::kInt);
  flx = evi * fl + (1 - evi) * flx;

  evi = (ev[4] < 0).to(torch::kInt);
  flx = evi * fr + (1 - evi) * flx;

  //--- Step 6.  Overwrite with LLF flux if any of intermediate states are
  // negative
  auto cmax =
      0.5 * torch::max(torch::abs(wl[ivx]) + cl, torch::abs(wr[ivx]) + cr);
  flx = llf_flag * (0.5 * (fl + fr) - cmax * du) + (1 - llf_flag) * flx;

  return flx;
}
}  // namespace snap
