#pragma once

// C/C++
#include <string>

// torch
#include <torch/torch.h>

namespace snap {

//! the x1 cell measure the corrected-PE work integrates over, and where x1v
//! sits in it (docs/derivations/curved-gravity-work-weight.md, secs 7, 12)
enum class X1Measure {
  none,       //!< no form for the corrected-PE work on this grid
  plain,      //!< dx1 (cartesian); x1v is the face midpoint, the centroid
  radial,     //!< r^2 dr (spherical-polar); x1v is the r^2 centroid
  radial_mid  //!< r^2 dr (gnomonic-equiangle); x1v is the face midpoint
};

inline X1Measure x1_measure(std::string const& type) {
  if (type == "cartesian") return X1Measure::plain;
  if (type == "spherical-polar") return X1Measure::radial;
  if (type == "gnomonic-equiangle") return X1Measure::radial_mid;
  return X1Measure::none;
}

//! <(x1 - r_c)^2> over each cell, r_c the centroid of the cell's own x1
//! measure, from its x1 faces (n + 1 of them): dx1^2/12 on a Cartesian grid,
//! the r^2 measure on a radial one (spherical-polar or gnomonic-equiangle;
//! r_c is x1v except on gnomonic-equiangle, docs/derivations/
//! curved-gravity-work-weight.md, secs 7, 12).
//! The r^2 form is written about the face midpoint so it does not cancel at
//! large r.
inline torch::Tensor x1_variance(torch::Tensor const& x1f, bool spherical) {
  int n = x1f.size(0) - 1;
  auto h = x1f.narrow(0, 1, n) - x1f.narrow(0, 0, n);
  auto h2 = h * h;
  if (!spherical) return h2 / 12.;
  auto rb = .5 * (x1f.narrow(0, 1, n) + x1f.narrow(0, 0, n));
  auto vol = rb * rb * h + h * h2 / 12.;  // (r+^3 - r-^3) / 3
  auto shift = rb * h * h2 / (6. * vol);  // r_c - rb
  return (rb * rb * h * h2 / 12. + h * h2 * h2 / 80.) / vol - shift * shift;
}

//! d q / d x1 at each x along the last dimension (n = x.size(0) cells): the
//! slope at x_i of the quadratic through cells i - 1, i, i + 1, and through
//! the first or last three cells at the two ends (at an internal x1 seam the
//! caller extends q by the neighbour's cell, corrected_pe_work_seam, so the
//! seam cell takes the centred slope; derivation sec 7). Zero for n < 3.
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

//! r_c - x1v for each cell, from its x1 faces (n + 1) and x1v (n): the
//! offset of the r^2 centroid r_c = rb + rb h^3 / (6 V_r) from x1v, written
//! about the face midpoint rb so it does not cancel at large r (sec 12)
inline torch::Tensor x1_centroid_offset(torch::Tensor const& x1f,
                                        torch::Tensor const& x1v) {
  int n = x1f.size(0) - 1;
  auto h = x1f.narrow(0, 1, n) - x1f.narrow(0, 0, n);
  auto h3 = h * h * h;
  auto rb = .5 * (x1f.narrow(0, 1, n) + x1f.narrow(0, 0, n));
  auto vol = rb * rb * h + h3 / 12.;  // (r+^3 - r-^3) / 3
  return rb * h3 / (6. * vol) + (rb - x1v);
}

//! x1 gravity work of the corrected potential energy beyond the face form,
//! per unit volume: grav1 [<(x1 - r_c)^2> d(drho)/dx1 + (r_c - x1v) drho],
//! for a density change drho of the interior cells [is, ie) (sec 7, eq. 7;
//! sec 12, eq. 12.5). r_c is the centroid of the cell measure; it is x1v
//! except on radial_mid, so the second term is booked only there. Booking it
//! makes E + P exact, P = sum V [rho phi(r_c) - grav1 <(x1 - r_c)^2> drho/dx1].
inline torch::Tensor corrected_pe_work(torch::Tensor const& drho,
                                       torch::Tensor const& x1f,
                                       torch::Tensor const& x1v, int is, int ie,
                                       double grav1, X1Measure measure) {
  auto faces = x1f.slice(0, is, ie + 1);
  auto var = x1_variance(faces, measure != X1Measure::plain);
  auto work = grav1 * var * centroid_slope(drho, x1v.slice(0, is, ie));
  if (measure == X1Measure::radial_mid)
    work += grav1 * x1_centroid_offset(faces, x1v.slice(0, is, ie)) * drho;
  return work;
}

//! corrected_pe_work of the interior cells [is, ie) when q, given over the
//! whole x1 array (..., nc1), also holds the neighbour's value one cell past an
//! internal x1 seam (seam_lo: cell is - 1, seam_hi: cell ie). The slope of a
//! seam cell is then the centred one that a single block takes, so the work
//! and P do not depend on the x1 decomposition; at a physical wall the slope
//! stays one-sided (sec 7 item 3, #303).
inline torch::Tensor corrected_pe_work_seam(torch::Tensor const& q,
                                            torch::Tensor const& x1f,
                                            torch::Tensor const& x1v, int is,
                                            int ie, double grav1,
                                            X1Measure measure, bool seam_lo,
                                            bool seam_hi) {
  int lo = seam_lo ? is - 1 : is, hi = seam_hi ? ie + 1 : ie;
  auto faces = x1f.slice(0, is, ie + 1);
  auto var = x1_variance(faces, measure != X1Measure::plain);
  auto s = centroid_slope(q.slice(-1, lo, hi), x1v.slice(0, lo, hi))
               .slice(-1, is - lo, ie - lo);
  auto work = grav1 * var * s;
  if (measure == X1Measure::radial_mid)
    work += grav1 * x1_centroid_offset(faces, x1v.slice(0, is, ie)) *
            q.slice(-1, is, ie);
  return work;
}

}  // namespace snap
