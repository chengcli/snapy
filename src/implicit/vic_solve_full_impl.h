#pragma once

// base
#include <configure.h>

// snap
#include "flux_decomposition_impl.h"
#include "forward_backward_impl.h"

#define GAMMA(n) gamma[(n) * stride2]
#define AREA(n) area[(n) * stride2]
#define VOL(n) vol[(n) * stride2]

namespace snap {

template <typename T>
void DISPATCH_MACRO vic_solve_full_impl(
    T *du, T *w, T *gamma, T *area, T *vol, double dt, double grav, int is,
    int ie, int dir, int ny, int stride1, int stride2, bool first_block,
    bool last_block, bool periodic, Eigen::Matrix<T, 5, 5> *a,
    Eigen::Matrix<T, 5, 5> *b, Eigen::Matrix<T, 5, 5> *c,
    Eigen::Matrix<T, 5, 1> *delta) {
  // eigenvectors, eigenvalues, inverse matrix of eigenvectors.
  Eigen::Matrix<T, 5, 5> Rmat, Lambda, Rimat;

  // reduced diffusion matrix |A_{i-1/2}|, |A_{i+1/2}|
  Eigen::Matrix<T, 5, 5> Am, Ap;
  Eigen::Matrix<T, 5, 5> dfdq[3];

  Eigen::Matrix<T, 5, 5> Phi, Dt, Bnd;
  Phi.setZero();
  Phi(IVX + dir, IDN) = grav;
  Phi(IPR, IVX + dir) = grav;

  Dt.setIdentity();
  Dt *= 1. / dt;

  Bnd.setIdentity();
  Bnd(IVX + dir, IVX + dir) = -1;

  T prim[5];       // Roe averaged primitive variables of cell i-1/2
  T wl[5], wr[5];  // left/right primitive variables of cell i-1 and i
  T gm1, cs;

  // 3. calculate and save flux Jacobian matrix
  for (int i = 0; i < 2; ++i) {
    int j = is - 1 + i;
    CopyPrimitives(wl, wr, w, j, stride1, stride2, ny);
    gm1 = GAMMA(j) - 1.;
    FluxJacobian(dfdq[i], gm1, wr, dir);
  }

  // 5. set up diffusion matrix and tridiagonal coefficients
  // left edge
  CopyPrimitives(wl, wr, w, is, stride1, stride2, ny);

  gm1 = 0.5 * (GAMMA(is - 1) + GAMMA(is)) - 1.;
  RoeAverage(prim, gm1, wl, wr);

  cs = SoundSpeed(prim, gm1);
  Eigenvalue(Lambda, prim[IVX + dir], cs);
  Eigenvector(Rmat, Rimat, prim, cs, gm1, dir);

  Am = Rmat * Lambda * Rimat;

  for (int i = is; i <= ie; ++i) {
    CopyPrimitives(wl, wr, w, i + 1, stride1, stride2, ny);
    gm1 = GAMMA(i + 1) - 1.;
    FluxJacobian(dfdq[2], gm1, wr, dir);

    // right edge
    gm1 = 0.5 * (GAMMA(i) + GAMMA(i + 1)) - 1.;
    RoeAverage(prim, gm1, wl, wr);

    cs = SoundSpeed(prim, gm1);
    Eigenvalue(Lambda, prim[IVX + dir], cs);
    Eigenvector(Rmat, Rimat, prim, cs, gm1, dir);

    Ap = Rmat * Lambda * Rimat;

    // set up diagonals a, b, c, and Jacobian of the forcing function
    a[i] =
        (Am * AREA(i) + Ap * AREA(i + 1) + (AREA(i + 1) - AREA(i)) * dfdq[1]) /
            (2. * VOL(i)) +
        Dt - Phi;
    b[i] = -(Am + dfdq[0]) * AREA(i) / (2. * VOL(i));
    c[i] = -(Ap - dfdq[2]) * AREA(i + 1) / (2. * VOL(i));

    // Shift one cell: i -> i+1
    Am = Ap;

    dfdq[0] = dfdq[1];
    dfdq[1] = dfdq[2];
  }

  // 5. fix boundary condition
  if (first_block && !periodic) a[is] += b[is] * Bnd;
  if (last_block && !periodic) a[ie] += c[ie] * Bnd;

  // 6. solve tridiagonal system
  if (periodic) {
    // PeriodicForwardSweep(a, b, c, corr, dt, is, ie);
  } else {
    ForwardSweep(a, b, c, delta, du, dt, is, ie, dir, ny, stride1, stride2,
                 first_block, last_block);
  }

  if (periodic) {
    // PeriodicBackwardSubstitution(a, c, delta, is, ie);
  } else {
    BackwardSubstitution(du, w, a, delta, is, ie, dir, ny, stride1, stride2,
                         first_block, last_block);
  }
}

}  // namespace snap

#undef GAMMA
#undef AREA
#undef VOL
