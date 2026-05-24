#pragma once

// eigen
#include <Eigen/Dense>

// base
#include <configure.h>

// snap
#include "flux_decomposition_impl.h"

#define GAMMA(n) gamma[(n) * stride2]
#define AREA(n) area[(n) * stride2]
#define VOL(n) vol[(n) * stride2]

namespace snap {

// Assemble the block-tridiagonal coefficients a[i], b[i], c[i] for a SINGLE
// vertical cell `i` of one column. This is the per-cell-parallel form of the
// assembly loop in vic_solve_partial_impl(): every cell is independent, so one
// GPU thread can own one cell. Shared interface Jacobians are recomputed from
// the same inputs, so the result is bit-identical to the serial assembly.
//
// a, b, c point to the start of the column's coefficient arrays (indexed [i]),
// matching the layout used by ForwardSweep/BackwardSubstitution.
template <typename T>
void DISPATCH_MACRO vic_assemble_partial_impl(
    Eigen::Matrix<T, 3, 3> *a, Eigen::Matrix<T, 3, 3> *b,
    Eigen::Matrix<T, 3, 3> *c, T *w, T *gamma, T *area, T *vol, int i, int is,
    int ie, double dt, double grav, int dir, int ny, int stride1, int stride2,
    bool first_block, bool last_block) {
  Eigen::Matrix<T, 5, 5> Rmat, Lambda, Rimat, Am, Ap, dfdqf;
  Eigen::Matrix<T, 3, 3> Am2, Ap2, dfdq_prev, dfdq_curr, dfdq_next;

  T prim[5];       // Roe averaged primitive variables at an interface
  T wl[5], wr[5];  // left/right primitive variables
  T gm1, cs;

  // ---- interface i-1/2 (cells i-1, i); also dfdq at cells i-1 and i ----
  CopyPrimitives(wl, wr, w, i, stride1, stride2,
                 ny);  // wl = cell i-1, wr = cell i

  gm1 = GAMMA(i - 1) - 1.;
  FluxJacobian(dfdqf, gm1, wl, dir);
  dfdq_prev << dfdqf(IDN, IDN), dfdqf(IDN, IVX), dfdqf(IDN, IPR),  //
      dfdqf(IVX, IDN), dfdqf(IVX, IVX), dfdqf(IVX, IPR),           //
      dfdqf(IPR, IDN), dfdqf(IPR, IVX), dfdqf(IPR, IPR);

  gm1 = GAMMA(i) - 1.;
  FluxJacobian(dfdqf, gm1, wr, dir);
  dfdq_curr << dfdqf(IDN, IDN), dfdqf(IDN, IVX), dfdqf(IDN, IPR),  //
      dfdqf(IVX, IDN), dfdqf(IVX, IVX), dfdqf(IVX, IPR),           //
      dfdqf(IPR, IDN), dfdqf(IPR, IVX), dfdqf(IPR, IPR);

  gm1 = 0.5 * (GAMMA(i - 1) + GAMMA(i)) - 1.;
  RoeAverage(prim, gm1, wl, wr);
  cs = SoundSpeed(prim, gm1);
  Eigenvalue(Lambda, prim[IVX + dir], cs);
  Eigenvector(Rmat, Rimat, prim, cs, gm1, dir);
  Am = Rmat * Lambda * Rimat;
  Am2 << Am(IDN, IDN), Am(IDN, IVX), Am(IDN, IPR),  //
      Am(IVX, IDN), Am(IVX, IVX), Am(IVX, IPR),     //
      Am(IPR, IDN), Am(IPR, IVX), Am(IPR, IPR);

  // ---- interface i+1/2 (cells i, i+1); also dfdq at cell i+1 ----
  CopyPrimitives(wl, wr, w, i + 1, stride1, stride2,
                 ny);  // wl = cell i, wr = cell i+1

  gm1 = GAMMA(i + 1) - 1.;
  FluxJacobian(dfdqf, gm1, wr, dir);
  dfdq_next << dfdqf(IDN, IDN), dfdqf(IDN, IVX), dfdqf(IDN, IPR),  //
      dfdqf(IVX, IDN), dfdqf(IVX, IVX), dfdqf(IVX, IPR),           //
      dfdqf(IPR, IDN), dfdqf(IPR, IVX), dfdqf(IPR, IPR);

  gm1 = 0.5 * (GAMMA(i) + GAMMA(i + 1)) - 1.;
  RoeAverage(prim, gm1, wl, wr);
  cs = SoundSpeed(prim, gm1);
  Eigenvalue(Lambda, prim[IVX + dir], cs);
  Eigenvector(Rmat, Rimat, prim, cs, gm1, dir);
  Ap = Rmat * Lambda * Rimat;
  Ap2 << Ap(IDN, IDN), Ap(IDN, IVX), Ap(IDN, IPR),  //
      Ap(IVX, IDN), Ap(IVX, IVX), Ap(IVX, IPR),     //
      Ap(IPR, IDN), Ap(IPR, IVX), Ap(IPR, IPR);

  // ---- set up diagonals a, b, c (matches vic_solve_partial_impl) ----
  a[i] = (Am2 * AREA(i) + Ap2 * AREA(i + 1) +
          (AREA(i + 1) - AREA(i)) * dfdq_curr) /
         (2. * VOL(i));
  // + Dt: add 1/dt to the diagonal
  a[i](0, 0) += 1. / dt;
  a[i](1, 1) += 1. / dt;
  a[i](2, 2) += 1. / dt;
  // - Phi: subtract gravity coupling
  a[i](1, 0) -= grav;
  a[i](2, 1) -= grav;
  b[i] = -(Am2 + dfdq_prev) * AREA(i) / (2. * VOL(i));
  c[i] = -(Ap2 - dfdq_next) * AREA(i + 1) / (2. * VOL(i));

  // ---- boundary condition. Bnd = diag(1, -1, 1) flips column 1 sign. ----
  if (i == is && first_block) {
    a[i].col(0) += b[i].col(0);
    a[i].col(1) -= b[i].col(1);
    a[i].col(2) += b[i].col(2);
  }
  if (i == ie && last_block) {
    a[i].col(0) += c[i].col(0);
    a[i].col(1) -= c[i].col(1);
    a[i].col(2) += c[i].col(2);
  }
}

}  // namespace snap

#undef GAMMA
#undef AREA
#undef VOL
