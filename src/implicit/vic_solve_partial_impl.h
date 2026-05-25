#pragma once

// base
#include <configure.h>

// snap
#include "flux_decomposition_impl.h"
#include "forward_backward_impl.h"
// #include "periodic_forward_backward_impl.h"

#define GAMMA(n) gamma[(n) * stride2]
#define AREA(n) area[(n) * stride2]
#define VOL(n) vol[(n) * stride2]

namespace snap {

template <typename T>
void DISPATCH_MACRO vic_solve_partial_impl(
    T* du, T* w, T* gamma, T* area, T* vol, double dt, double grav, int is,
    int ie, int dir, int ny, int stride1, int stride2, bool first_block,
    bool last_block, bool periodic, Eigen::Matrix<T, 3, 3>* a,
    Eigen::Matrix<T, 3, 3>* b, Eigen::Matrix<T, 3, 3>* c,
    Eigen::Matrix<T, 3, 1>* delta) {
  // eigenvectors, eigenvalues, inverse matrix of eigenvectors.
  Eigen::Matrix<T, 5, 5> Rmat, Rimat;
  Eigen::Matrix<T, 5, 1> Lambda;

  // reduced diffusion matrix |A_{i-1/2}|, |A_{i+1/2}|
  Eigen::Matrix<T, 5, 5> Am, Ap, dfdqf;
  Eigen::Matrix<T, 3, 2> Am1, Ap1;
  Eigen::Matrix<T, 3, 3> Am2, Ap2;
  Eigen::Matrix<T, 3, 3> dfdq[3];

  Eigen::Matrix<T, 3, 3> Phi;
  Eigen::Matrix<T, 3, 1> Dt, Bnd;

  Phi << 0., 0., 0.,  //
      grav, 0., 0.,   //
      0., grav, 0.;

  Dt << 1. / dt, 1. / dt, 1. / dt;
  Bnd << 1., -1, 1.;

  T prim[5];       // Roe averaged primitive variables of cell i-1/2
  T wl[5], wr[5];  // left/right primitive variables of cell i-1 and i
  T gm1, cs;

  // calculate and save flux Jacobian matrix
  for (int i = 0; i < 2; ++i) {
    int j = is - 1 + i;
    CopyPrimitives(wl, wr, w, j, stride1, stride2, ny);
    gm1 = GAMMA(j) - 1.;
    FluxJacobian(dfdqf, gm1, wr, dir);

    dfdq[i] << dfdqf(IDN, IDN), dfdqf(IDN, IVX), dfdqf(IDN, IPR),  //
        dfdqf(IVX, IDN), dfdqf(IVX, IVX), dfdqf(IVX, IPR),         //
        dfdqf(IPR, IDN), dfdqf(IPR, IVX), dfdqf(IPR, IPR);
  }

  // left edge
  CopyPrimitives(wl, wr, w, is, stride1, stride2, ny);

  gm1 = 0.5 * (GAMMA(is - 1) + GAMMA(is)) - 1.;
  RoeAverage(prim, gm1, wl, wr);

  cs = SoundSpeed(prim, gm1);
  Eigenvalue(Lambda, prim[IVX + dir], cs);
  Eigenvector(Rmat, Rimat, prim, cs, gm1, dir);

  Am.noalias() = Rmat * Lambda.asDiagonal() * Rimat;

  Am1 << Am(IDN, IVY), Am(IDN, IVZ), Am(IVX, IVY),  //
      Am(IVX, IVZ), Am(IPR, IVY), Am(IPR, IVZ);

  Am2 << Am(IDN, IDN), Am(IDN, IVX), Am(IDN, IPR),  //
      Am(IVX, IDN), Am(IVX, IVX), Am(IVX, IPR),     //
      Am(IPR, IDN), Am(IPR, IVX), Am(IPR, IPR);

  for (int i = is; i <= ie; ++i) {
    CopyPrimitives(wl, wr, w, i + 1, stride1, stride2, ny);
    gm1 = GAMMA(i + 1) - 1.;
    FluxJacobian(dfdqf, gm1, wr, dir);

    dfdq[2] << dfdqf(IDN, IDN), dfdqf(IDN, IVX), dfdqf(IDN, IPR),  //
        dfdqf(IVX, IDN), dfdqf(IVX, IVX), dfdqf(IVX, IPR),         //
        dfdqf(IPR, IDN), dfdqf(IPR, IVX), dfdqf(IPR, IPR);

    gm1 = 0.5 * (GAMMA(i) + GAMMA(i + 1)) - 1.;
    RoeAverage(prim, gm1, wl, wr);

    cs = SoundSpeed(prim, gm1);
    Eigenvalue(Lambda, prim[IVX + dir], cs);
    Eigenvector(Rmat, Rimat, prim, cs, gm1, dir);

    Ap.noalias() = Rmat * Lambda.asDiagonal() * Rimat;

    Ap1 << Ap(IDN, IVY), Ap(IDN, IVZ), Ap(IVX, IVY),  //
        Ap(IVX, IVZ), Ap(IPR, IVY), Ap(IPR, IVZ);

    Ap2 << Ap(IDN, IDN), Ap(IDN, IVX), Ap(IDN, IPR),  //
        Ap(IVX, IDN), Ap(IVX, IVX), Ap(IVX, IPR),     //
        Ap(IPR, IDN), Ap(IPR, IVX), Ap(IPR, IPR);

    T const& area_i = AREA(i);
    T const& area_ip1 = AREA(i + 1);
    T half_inv_vol = 0.5 / VOL(i);

    // set up diagonals a, b, c.
    a[i] = (Am2 * area_i + Ap2 * area_ip1 + (area_ip1 - area_i) * dfdq[1]) *
               half_inv_vol -
           Phi;
    a[i].diagonal() += Dt;
    b[i] = -(Am2 + dfdq[0]) * area_i * half_inv_vol;
    c[i] = -(Ap2 - dfdq[2]) * area_ip1 * half_inv_vol;

    // Shift one cell: i -> i+1
    Am1 = Ap1;
    Am2 = Ap2;

    dfdq[0] = dfdq[1];
    dfdq[1] = dfdq[2];
  }

  // 5. fix boundary condition
  if (first_block && !periodic) a[is] += b[is] * Bnd.asDiagonal();
  if (last_block && !periodic) a[ie] += c[ie] * Bnd.asDiagonal();

  // 6. solve tridiagonal system using LU decomposition
  if (periodic) {
    // PeriodicForwardSweep(a, b, c, dt, is, ie);
  } else {
    ForwardSweep(a, b, c, delta, du, dt, is, ie, dir, ny, stride1, stride2,
                 first_block, last_block);
  }

  if (periodic) {
    // PeriodicBackwardSubstitution(a, c, delta, is, ie);
  } else {
    BackwardSubstitution(du, w, a, delta, vol, is, ie, dir, ny, stride1,
                         stride2, first_block, last_block);
  }
}

}  // namespace snap

#undef GAMMA
#undef AREA
#undef VOL
