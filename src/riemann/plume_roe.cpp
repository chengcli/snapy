// snap
#include <snap/snap.h>

#include <snap/hydro/hydro.hpp>

#include "riemann_solver.hpp"

namespace snap {

//! calculate flux for plume equations
/*!
 * w = [R, W, B, V]
 */
static void calculate_flux(torch::Tensor& flx, torch::Tensor const& prim) {
  auto R = prim[0];
  auto W = prim[1];
  auto B = prim[2];
  auto V = prim[3];

  flx[0] = (1.0 - 1.0 / std::exp(1.0)) * R * R * W;
  flx[1] = 0.5 * R * R * W * V - 0.5 * R * R * R * V - 0.125 * R * R * V * V;
  flx[2] = 0.5 * R * R * B * W;
  flx[3] = 0.25 * R * R * R * V * W + 0.5 * R * R * R * R * W;
}

static torch::Tensor lax_friedrichs_flux(torch::Tensor const& priml,
                                         torch::Tensor const& primr,
                                         torch::Tensor const& csl,
                                         torch::Tensor const& csr) {
  auto flxl = torch::zeros_like(priml);
  auto flxr = torch::zeros_like(primr);

  calculate_flux(flxl, priml);
  calculate_flux(flxr, primr);

  auto alpha = torch::max(csl, csr);

  return 0.5 * (flxl + flxr) - 0.5 * alpha * (primr - priml);
}

void PlumeRoeSolverImpl::reset() {
  TORCH_CHECK(phydro, "[PlumeRoeSolver] parent is nullptr");
}

torch::Tensor PlumeRoeSolverImpl::forward(torch::Tensor wl, torch::Tensor wr,
                                          int dim, torch::Tensor flx) {
  auto peos = phydro->peos;

  if (dim != 1) {
    TORCH_CHECK(false,
                "PlumeRoeSolver only supports dim = 1, but got dim = ", dim);
  }

  // Left-right states
  auto Rl = wl[0];
  auto Wl = wl[1];
  auto Bl = wl[2];
  auto Vl = wl[3];

  auto Rr = wr[0];
  auto Wr = wr[1];
  auto Br = wr[2];
  auto Vr = wr[3];

  // Roe averages
  auto sqrtRl = Rl.sqrt();
  auto sqrtRr = Rr.sqrt();
  auto isqrtR = 1.0 / (sqrtRl + sqrtRr);

  auto R_avg = sqrtRl * sqrtRr;
  auto W_avg = (sqrtRl * Wl + sqrtRr * Wr) * isqrtR;
  auto B_avg = (sqrtRl * Bl + sqrtRr * Br) * isqrtR;
  auto V_avg = (sqrtRl * Vl + sqrtRr * Vr) * isqrtR;

  auto c1 = 1. - 1. / std::exp(1.0);

  // Jacobians
  auto J21 = -0.5 * W_avg * W_avg + 0.25 * V_avg * V_avg;
  auto J22 = W_avg;
  auto J24 = -0.5 - 0.25 * (V_avg / R_avg);

  auto J41 = 0.5 * (R_avg * R_avg) * W_avg - 0.25 * R_avg * W_avg * V_avg;
  auto J42 = 0.5 * (R_avg * R_avg) + 0.25 * R_avg * V_avg;
  auto J44 = 0.25 * W_avg;

  // Characteristic Polynomial: lambda^3 + a2*lambda^2 + a1*lambda + a0 = 0
  auto a2 = -(J22 + J44);
  auto a1 = (J22 * J44 - J24 * J42) - c1 * J21;
  auto a0 = c1 * (J21 * J44 - J24 * J41);

  // Approximate sound speeds
  auto csl = peos->compute("W->L", {wl});
  auto csr = peos->compute("W->L", {wr});

  // Lax-Friedrichs flux
  auto lf_flux = lax_friedrichs_flux(wl, wr, csl, csr);

  return lf_flux;
}

}  // namespace snap
