#pragma once

// torch
#include <torch/torch.h>

namespace snap {

//! <(x1 - x1v)^2> over each cell, from its x1 faces (n + 1 of them), on the
//! cell's own x1 measure: dx1^2/12 on a Cartesian grid, the r^2 measure on a
//! spherical-polar one (docs/derivations/curved-gravity-work-weight.md, sec 7).
//! The r^2 form is written about the face midpoint so it does not cancel at
//! large r.
inline torch::Tensor x1_variance(torch::Tensor const& x1f, bool spherical) {
  int n = x1f.size(0) - 1;
  auto h = x1f.narrow(0, 1, n) - x1f.narrow(0, 0, n);
  auto h2 = h * h;
  if (!spherical) return h2 / 12.;
  auto rb = .5 * (x1f.narrow(0, 1, n) + x1f.narrow(0, 0, n));
  auto vol = rb * rb * h + h * h2 / 12.;  // (r+^3 - r-^3) / 3
  auto shift = rb * h * h2 / (6. * vol);  // x1v - rb
  return (rb * rb * h * h2 / 12. + h * h2 * h2 / 80.) / vol - shift * shift;
}

//! d q / d x1 at each x along the last dimension (n = x.size(0) cells): the
//! slope at x_i of the quadratic through cells i - 1, i, i + 1, and through
//! the first or last three cells at the two ends. Zero for n < 3.
inline torch::Tensor centroid_slope(torch::Tensor const& q,
                                    torch::Tensor const& x) {
  int n = x.size(0);
  auto s = torch::zeros_like(q);
  if (n < 3) return s;
  auto hm = x.narrow(0, 1, n - 2) - x.narrow(0, 0, n - 2);
  auto hp = x.narrow(0, 2, n - 2) - x.narrow(0, 1, n - 2);
  s.narrow(-1, 1, n - 2)
      .copy_(-hp / (hm * (hm + hp)) * q.narrow(-1, 0, n - 2) +
             (hp - hm) / (hm * hp) * q.narrow(-1, 1, n - 2) +
             hm / (hp * (hm + hp)) * q.narrow(-1, 2, n - 2));
  auto a = x.narrow(0, 1, 1) - x.narrow(0, 0, 1);
  auto b = x.narrow(0, 2, 1) - x.narrow(0, 1, 1);
  s.narrow(-1, 0, 1).copy_(-(2. * a + b) / (a * (a + b)) * q.narrow(-1, 0, 1) +
                           (a + b) / (a * b) * q.narrow(-1, 1, 1) -
                           a / ((a + b) * b) * q.narrow(-1, 2, 1));
  a = x.narrow(0, n - 2, 1) - x.narrow(0, n - 3, 1);
  b = x.narrow(0, n - 1, 1) - x.narrow(0, n - 2, 1);
  s.narrow(-1, n - 1, 1)
      .copy_(b / (a * (a + b)) * q.narrow(-1, n - 3, 1) -
             (a + b) / (a * b) * q.narrow(-1, n - 2, 1) +
             (a + 2. * b) / ((a + b) * b) * q.narrow(-1, n - 1, 1));
  return s;
}

//! x1 gravity work of the corrected potential energy beyond the face form,
//! per unit volume: grav1 <(x1 - x1v)^2> d(drho)/dx1, for a density change
//! drho of the interior cells [is, ie) (sec 7, eq. 7). Booking it makes
//! E + P exact, P = sum V [rho phi(x1v) - grav1 <(x1 - x1v)^2> drho/dx1].
inline torch::Tensor corrected_pe_work(torch::Tensor const& drho,
                                       torch::Tensor const& x1f,
                                       torch::Tensor const& x1v, int is, int ie,
                                       double grav1, bool spherical) {
  auto var = x1_variance(x1f.slice(0, is, ie + 1), spherical);
  return grav1 * var * centroid_slope(drho, x1v.slice(0, is, ie));
}

}  // namespace snap
