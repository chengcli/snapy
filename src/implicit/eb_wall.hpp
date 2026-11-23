#pragma once

#include <snap/snap.h>
#include <torch/torch.h>

namespace snap {

// Linearize the symmetric stationary-slip Riemann problem at the reconstructed
// wall state. The centroid reconstruction offsets, composition and gamma are
// frozen, like the Roe eigenbasis in the vertical solve. Inputs and the five
// derivatives are in Cartesian order (rho_mix, m1, m2, m3, E_without_offsets).
inline torch::Tensor wall_pressure_gradient(torch::Tensor const& wall,
                                            torch::Tensor const& center,
                                            torch::Tensor const& normal,
                                            torch::Tensor const& gamma_wall,
                                            torch::Tensor const& gamma_cell,
                                            bool hllc) {
  auto rho = wall[IDN], p = wall[IPR];
  auto un = (wall.narrow(0, IVX, 3) * normal).sum(0);
  auto impedance = (gamma_wall * p * rho).sqrt();
  auto y = impedance * un;
  auto chi = y / p;
  auto coefficient =
      hllc ? (gamma_wall + 1.) / (2. * gamma_wall) : torch::zeros_like(p);
  auto q = (1. + coefficient * chi.clamp_min(0.)).sqrt();
  auto dq = torch::where(chi > 0., coefficient / (2. * q), 0.);
  auto fy = q + chi * dq;
  auto fp = 1. - chi.square() * dq + .5 * fy * y / p;
  auto vel = center.narrow(0, IVX, 3);
  auto g = fp * (gamma_cell - 1.);
  auto fv = fy * impedance / center[IDN];
  auto dr = .5 * g * vel.square().sum(0) + .5 * fy * y / rho -
            fv * (vel * normal).sum(0);
  auto dm = fv * normal - g * vel;
  auto grad = torch::cat({dr.unsqueeze(0), dm, g.unsqueeze(0)});
  return hllc ? torch::where(p + q * y > 0., grad, 0.) : grad;
}

// The momentum flux Jacobian left by VIC's reflected Roe flux through the
// lower x1 domain boundary. W carries the interior state; its ghost reflects
// m1. The Roe density is rho and its sound speed includes normal kinetic heat.
inline torch::Tensor reflecting_lower_pressure_increment(torch::Tensor w,
                                                         torch::Tensor gamma,
                                                         torch::Tensor delta) {
  auto v = w.narrow(0, IVX, 3), m = delta.narrow(0, IVX, 3);
  auto dp = (gamma - 1.) *
            (delta[IPR] - (v * m).sum(0) + .5 * v.square().sum(0) * delta[IDN]);
  auto c =
      (gamma * w[IPR] / w[IDN] + .5 * (gamma - 1.) * w[IVX].square()).sqrt();
  return dp + (2. * w[IVX] - c) * delta[IVX] - w[IVX].square() * delta[IDN];
}
}  // namespace snap
