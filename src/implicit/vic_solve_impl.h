#pragma once

// snap
#include "flux_decomposition_impl.h"
#include "forward_backward_impl.h"
// #include "periodic_forward_backward_impl.h"

namespace snap {

template <typename T>
T SoundSpeed(T *prim, T gm1) {
  return sqrt(prim[IPR] * (gm1 + 1.) / prim[IDN]);
}

template <typename T>
void vic_partial_solve_impl(
    T *du, T *w, T *gamma, T *area, T *vol, double dt, double grav, int is,
    int ie, int dir, int ny, int stride1, int stride2, bool first_block,
    bool last_block, bool periodic, Eigen::Matrix<T, 3, 3> *a,
    Eigen::Matrix<T, 3, 3> *b, Eigen::Matrix<T, 3, 3> *c,
    Eigen::Matrix<T, 3, 1> *delta, Eigen::Matrix<T, 3, 3> *dfdq) {
  // reduced diffusion matrix |A_{i-1/2}|, |A_{i+1/2}|
  Eigen::Matrix<T, 5, 5> Am, Ap, dfdq1;
  Eigen::Matrix<T, 3, 2> Am1, Ap1;
  Eigen::Matrix<T, 3, 3> Am2, Ap2;

  // int nc = ie - is + 1 + 2 * NGHOST;

  Eigen::Matrix<T, 3, 3> Phi, Dt, Bnd;

  Phi << 0., 0., 0.,  //
      grav, 0., 0.,   //
      0., grav, 0.;

  Dt << 1. / dt, 0., 0.,  //
      0., 1. / dt, 0.,    //
      0., 0., 1. / dt;

  Bnd << 1., 0., 0.,  //
      0., -1., 0.,    //
      0., 0., 1.;

  T prim[5];       // Roe averaged primitive variables of cell i-1/2
  T wl[5], wr[5];  // left/right primitive variables of cell i-1 and i

  // calculate and save flux Jacobian matrix
  for (int i = is - 2; i <= ie + 1; ++i) {
    CopyPrimitives(wl, wr, w, i, stride1, stride2);
    FluxJacobian(dfdq1, static_cast<T>(gamma[i * stride2] - 1.), wr, dir);

    dfdq[i] << dfdq1(IDN, IDN), dfdq1(IDN, IVX), dfdq1(IDN, IPR),  //
        dfdq1(IVX, IDN), dfdq1(IVX, IVX), dfdq1(IVX, IPR),         //
        dfdq1(IPR, IDN), dfdq1(IPR, IVX), dfdq1(IPR, IPR);
  }

  // set up diffusion matrix and tridiagonal coefficients
  // eigenvectors, eigenvalues, inverse matrix of eigenvectors.
  Eigen::Matrix<T, 5, 5> Rmat, Lambda, Rimat;

  // left edge
  CopyPrimitives(wl, wr, w, is - 1, stride1, stride2);
  T gm1 = 0.5 * (gamma[(is - 2) * stride2] + gamma[(is - 1) * stride2]) - 1.;
  RoeAverage(prim, gm1, wl, wr);
  T cs = SoundSpeed(prim, gm1);
  Eigenvalue(Lambda, prim[IVX + dir], cs);
  Eigenvector(Rmat, Rimat, prim, cs, gm1, dir);
  Am = Rmat * Lambda * Rimat;

  Am1 << Am(IDN, IVY), Am(IDN, IVZ), Am(IVX, IVY),  //
      Am(IVX, IVZ), Am(IPR, IVY), Am(IPR, IVZ);

  Am2 << Am(IDN, IDN), Am(IDN, IVX), Am(IDN, IPR),  //
      Am(IVX, IDN), Am(IVX, IVX), Am(IVX, IPR),     //
      Am(IPR, IDN), Am(IPR, IVX), Am(IPR, IPR);

  for (int i = is - 1; i <= ie; ++i) {
    CopyPrimitives(wl, wr, w, i + 1, stride1, stride2);
    gm1 = 0.5 * (gamma[i * stride2] + gamma[(i + 1) * stride2]) - 1.;
    RoeAverage(prim, gm1, wl, wr);
    T cs = SoundSpeed(prim, gm1);
    Eigenvalue(Lambda, prim[IVX + dir], cs);
    Eigenvector(Rmat, Rimat, prim, cs, gm1, dir);
    Ap = Rmat * Lambda * Rimat;

    Ap1 << Ap(IDN, IVY), Ap(IDN, IVZ), Ap(IVX, IVY),  //
        Ap(IVX, IVZ), Ap(IPR, IVY), Ap(IPR, IVZ);

    Ap2 << Ap(IDN, IDN), Ap(IDN, IVX), Ap(IDN, IPR),  //
        Ap(IVX, IDN), Ap(IVX, IVX), Ap(IVX, IPR),     //
        Ap(IPR, IDN), Ap(IPR, IVX), Ap(IPR, IPR);

    // set up diagonals a, b, c.
    a[i] = (Am2 * area[i] + Ap2 * area[i + 1] +
            (area[i + 1] - area[i]) * dfdq[i]) /
               (2. * vol[i]) +
           Dt - Phi;
    b[i] = -(Am2 + dfdq[i - 1]) * area[i] / (2. * vol[i]);
    c[i] = -(Ap2 - dfdq[i + 1]) * area[i + 1] / (2. * vol[i]);

    // Shift one cell: i -> i+1
    Am1 = Ap1;
    Am2 = Ap2;
  }

  // 5. fix boundary condition
  if (first_block && !periodic) a[is] += b[is] * Bnd;
  if (last_block && !periodic) a[ie] += c[ie] * Bnd;

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
    BackwardSubstitution(du, w, a, delta, is, ie, dir, ny, stride1, stride2,
                         first_block, last_block);
  }
}

}  // namespace snap
