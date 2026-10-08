#pragma once

// eigen
#include <Eigen/Dense>

// base
#include <configure.h>

// snap
#include "flux_decomposition_impl.h"
#include "implicit_dispatch.hpp"

#define GAMMA(n) gamma[(n) * stride2]
#define AREA(n) area[(n) * stride2]
#define VOL(n) vol[(n) * stride2]

namespace snap {

// Assemble the block-tridiagonal coefficients a[i], b[i], c[i] for a SINGLE
// vertical cell `i` of one column. This is the per-cell-parallel form of the
// assembly loop in vic_solve_partial_impl(): every cell is independent, so one
// GPU thread can own one cell. Shared interface Jacobians are recomputed from
// the same inputs, so the result matches the serial assembly. Kept in lockstep
// with vic_solve_partial_impl() so the GPU (split) and CPU (fused) paths agree.
//
// a, b, c point to the start of the column's coefficient arrays (indexed [i]),
// matching the layout used by ForwardSweep and the split redistribution
// helpers.
template <typename T>
void DISPATCH_MACRO vic_assemble_partial_impl(
    Eigen::Matrix<T, 3, 3>* a, Eigen::Matrix<T, 3, 3>* b,
    Eigen::Matrix<T, 3, 3>* c, T* w, T* gamma, T* area, T* vol, T* work_lo,
    T* work_hi, int i, int is, int ie, double dt, double grav, int dir, int ny,
    int stride1, int stride2, bool first_block, bool last_block,
    bool solid_lower = false, bool solid_upper = false) {
  bool face_work = dir & kVicFaceWork;
  bool diffusive_cell = dir & kVicDiffusiveCell;
  dir &= ~(kVicFaceWork | kVicDiffusiveCell);
  Eigen::Matrix<T, 5, 5> Rmat, Rimat, Am, Ap, dfdqf;
  Eigen::Matrix<T, 5, 1> Lambda;
  Eigen::Matrix<T, 3, 3> Am2, Ap2, dfdq_prev, dfdq_curr, dfdq_next;

  Eigen::Matrix<T, 3, 3> Phi;
  Eigen::Matrix<T, 3, 1> Dt, Bnd;
  Phi << 0., 0., 0.,  //
      grav, 0., 0.,   //
      0., grav, 0.;
  Dt << 1. / dt, 1. / dt, 1. / dt;
  Bnd << 1., -1, 1.;

  T prim[5];       // Roe averaged primitive variables at an interface
  T wl[5], wr[5];  // left/right primitive variables
  T gm1, cs;

  // ---- interface i-1/2 (cells i-1, i); also dfdq at cells i-1 and i ----
  CopyPrimitives(wl, wr, w, i, stride1, stride2,
                 ny);  // wl = cell i-1, wr = cell i

  if (solid_lower) {
    for (int n = 0; n < 5; ++n) wl[n] = wr[n];
    wl[IVX + dir] = -wr[IVX + dir];
  }
  gm1 = GAMMA(solid_lower ? i : i - 1) - 1.;
  FluxJacobian(dfdqf, gm1, wl, dir);
  dfdq_prev << dfdqf(IDN, IDN), dfdqf(IDN, IVX), dfdqf(IDN, IPR),  //
      dfdqf(IVX, IDN), dfdqf(IVX, IVX), dfdqf(IVX, IPR),           //
      dfdqf(IPR, IDN), dfdqf(IPR, IVX), dfdqf(IPR, IPR);

  gm1 = GAMMA(i) - 1.;
  FluxJacobian(dfdqf, gm1, wr, dir);
  dfdq_curr << dfdqf(IDN, IDN), dfdqf(IDN, IVX), dfdqf(IDN, IPR),  //
      dfdqf(IVX, IDN), dfdqf(IVX, IVX), dfdqf(IVX, IPR),           //
      dfdqf(IPR, IDN), dfdqf(IPR, IVX), dfdqf(IPR, IPR);

  gm1 = 0.5 * (GAMMA(solid_lower ? i : i - 1) + GAMMA(i)) - 1.;
  RoeAverage(prim, gm1, wl, wr);
  cs = SoundSpeed(prim, gm1);
  Eigenvalue(Lambda, prim[IVX + dir], cs);
  Eigenvector(Rmat, Rimat, prim, cs, gm1, dir);
  Am.noalias() = Rmat * Lambda.asDiagonal() * Rimat;
  Am2 << Am(IDN, IDN), Am(IDN, IVX), Am(IDN, IPR),  //
      Am(IVX, IDN), Am(IVX, IVX), Am(IVX, IPR),     //
      Am(IPR, IDN), Am(IPR, IVX), Am(IPR, IPR);

  // ---- interface i+1/2 (cells i, i+1); also dfdq at cell i+1 ----
  CopyPrimitives(wl, wr, w, i + 1, stride1, stride2,
                 ny);  // wl = cell i, wr = cell i+1

  if (solid_upper) {
    for (int n = 0; n < 5; ++n) wr[n] = wl[n];
    wr[IVX + dir] = -wl[IVX + dir];
  }
  gm1 = GAMMA(solid_upper ? i : i + 1) - 1.;
  FluxJacobian(dfdqf, gm1, wr, dir);
  dfdq_next << dfdqf(IDN, IDN), dfdqf(IDN, IVX), dfdqf(IDN, IPR),  //
      dfdqf(IVX, IDN), dfdqf(IVX, IVX), dfdqf(IVX, IPR),           //
      dfdqf(IPR, IDN), dfdqf(IPR, IVX), dfdqf(IPR, IPR);

  gm1 = 0.5 * (GAMMA(i) + GAMMA(solid_upper ? i : i + 1)) - 1.;
  RoeAverage(prim, gm1, wl, wr);
  cs = SoundSpeed(prim, gm1);
  Eigenvalue(Lambda, prim[IVX + dir], cs);
  Eigenvector(Rmat, Rimat, prim, cs, gm1, dir);
  Ap.noalias() = Rmat * Lambda.asDiagonal() * Rimat;
  Ap2 << Ap(IDN, IDN), Ap(IDN, IVX), Ap(IDN, IPR),  //
      Ap(IVX, IDN), Ap(IVX, IVX), Ap(IVX, IPR),     //
      Ap(IPR, IDN), Ap(IPR, IVX), Ap(IPR, IPR);

  // ---- set up diagonals a, b, c (matches vic_solve_partial_impl) ----
  T const& area_i = AREA(i);
  T const& area_ip1 = AREA(i + 1);
  T half_inv_vol = 0.5 / VOL(i);

  a[i] = (Am2 * area_i + Ap2 * area_ip1 + (area_ip1 - area_i) * dfdq_curr) *
             half_inv_vol -
         Phi;
  a[i].diagonal() += Dt;
  b[i] = -(Am2 + dfdq_prev) * area_i * half_inv_vol;
  c[i] = -(Ap2 - dfdq_next) * area_ip1 * half_inv_vol;

  if (face_work || diffusive_cell) {
    // The same face-to-centre potential weights as the mass update.
    T lower = work_lo[i * stride2], upper = work_hi[i * stride2];
    a[i].row(2) -= grav * (upper * Ap2.row(0) - lower * Am2.row(0));
    b[i].row(2) -= grav * lower * Am2.row(0);
    c[i].row(2) += grav * upper * Ap2.row(0);
    if (face_work) {
      a[i](2, 1) += grav * (1. - lower - upper);
      b[i](2, 1) -= grav * lower;
      c[i](2, 1) -= grav * upper;
    }
  }

  // ---- boundary condition. Bnd = diag(1, -1, 1). ----
  if ((i == is || solid_lower) && first_block) a[i] += b[i] * Bnd.asDiagonal();
  if ((i == ie || solid_upper) && last_block) a[i] += c[i] * Bnd.asDiagonal();
}

}  // namespace snap

#undef GAMMA
#undef AREA
#undef VOL
