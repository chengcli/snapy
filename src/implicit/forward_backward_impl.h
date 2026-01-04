#pragma once

// eigen
#include <Eigen/Dense>

// base
#include <configure.h>

// snap
#include <snap/math/ludcmp.h>
#include <snap/math/luminv.h>
#include <snap/snap.h>

#define DU(n, i) du[(n) * stride1 + (i) * stride2]
#define W(n, i) w[(n) * stride1 + (i) * stride2]

namespace snap {

template <typename T, int N>
void DISPATCH_MACRO ForwardSweep(Eigen::Matrix<T, N, N> *a,
                                 Eigen::Matrix<T, N, N> *b,
                                 Eigen::Matrix<T, N, N> *c,
                                 Eigen::Matrix<T, N, 1> *delta, T *du,
                                 double dt, int il, int iu, int dir, int ny,
                                 int stride1, int stride2, bool first_block,
                                 bool last_block) {
  Eigen::Matrix<T, N, 1> rhs;

  if constexpr (N == 3) {  // partial matrix
    rhs(0) = DU(IDN, il);
    for (int n = 0; n < ny; ++n) rhs(0) += DU(ICY + n, il);
    rhs(0) /= dt;
    rhs(1) = DU(IVX + dir, il) / dt;
    rhs(2) = DU(IPR, il) / dt;
  } else {  // full matrix
    rhs(0) = DU(IDN, il);
    for (int n = 0; n < ny; ++n) rhs(0) += DU(ICY + n, il);
    rhs(0) /= dt;
    rhs(1) = DU(IVX + dir, il) / dt;
    rhs(2) = DU(IVX + (IVY - IVX + dir) % 3, il) / dt;
    rhs(3) = DU(IVX + (IVZ - IVX + dir) % 3, il) / dt;
    rhs(4) = DU(IPR, il) / dt;
  }

  int indx[N];
  Eigen::Matrix<T, N, N, Eigen::RowMajor> A, Y;

  // if (!first_block) {
  // RecvBuffer(a[il - 1], delta[il - 1], bblock);
  // a[il] = (a[il] - b[il] * a[il - 1]).inverse().eval();
  // delta[il] = a[il] * (rhs - b[il] * delta[il - 1]);
  // a[il] *= c[il];
  //} else {
  if constexpr (N > 4) {
    A = a[il].transpose();
    for (int n = 0; n < N; ++n) indx[n] = n;
    ludcmp(A, indx);
    luminv(A, indx, Y);
    a[il] = Y.transpose();
  } else {  // small matrix
    a[il] = a[il].inverse().eval();
  }

  delta[il] = a[il] * rhs;
  a[il] = a[il] * c[il];
  //}

  for (int i = il + 1; i <= iu; ++i) {
    if constexpr (N == 3) {  // partial matrix
      rhs(0) = DU(IDN, i);
      for (int n = 0; n < ny; ++n) rhs(0) += DU(ICY + n, i);
      rhs(0) /= dt;
      rhs(1) = DU(IVX + dir, i) / dt;
      rhs(2) = DU(IPR, i) / dt;
    } else {
      rhs(0) = DU(IDN, i);
      for (int n = 0; n < ny; ++n) rhs(0) += DU(ICY + n, i);
      rhs(0) /= dt;
      rhs(1) = DU(IVX + dir, i) / dt;
      rhs(2) = DU(IVX + (IVY - IVX + dir) % 3, i) / dt;
      rhs(3) = DU(IVX + (IVZ - IVX + dir) % 3, i) / dt;
      rhs(4) = DU(IPR, i) / dt;
    }

    a[i] -= b[i] * a[i - 1];

    if constexpr (N > 4) {
      A = a[i].transpose();
      for (int n = 0; n < N; ++n) indx[n] = n;
      ludcmp(A, indx);
      luminv(A, indx, Y);
      a[i] = Y.transpose();
    } else {  // small matrix
      a[i] = a[i].inverse().eval();
    }

    delta[i] = a[i] * (rhs - b[i] * delta[i - 1]);
    a[i] = a[i] * c[i];
  }

  // SaveCoefficients(a, delta, il, iu);
  // if (!last_block) SendBuffer(a[iu], delta[iu], tblock);
}

template <typename T, int N>
void DISPATCH_MACRO BackwardSubstitution(T *du, T *w, Eigen::Matrix<T, N, N> *a,
                                         Eigen::Matrix<T, N, 1> *delta, int il,
                                         int iu, int dir, int ny, int stride1,
                                         int stride2, bool first_block,
                                         bool last_block) {
  // LoadCoefficients(a, delta, il, iu);
  // if (!last_block) {
  //   RecvBuffer(delta[iu + 1], tblock);
  //   delta[iu] -= a[iu] * delta[iu + 1];
  // }

  // update solutions, i=iu
  for (int i = iu - 1; i >= il; --i) delta[i] -= a[i] * delta[i + 1];

  // 7. update conserved variables, i = iu
  for (int i = il; i <= iu; ++i) {
    T dens = DU(IDN, i);
    for (int n = 0; n < ny; ++n) dens += DU(ICY + n, i);
    dens = delta[i](0) - dens;

    if constexpr (N == 3) {  // partial matrix
      DU(IDN, i) = delta[i](0);
      for (int n = 0; n < ny; ++n) {
        DU(ICY + n, i) += dens * W(ICY + n, i);
        DU(IDN, i) -= dens * W(ICY + n, i);
      }
      DU(IVX + dir, i) = delta[i](1);
      DU(IPR, i) = delta[i](2);
    } else {  // full matrix
      DU(IDN, i) = delta[i](0);
      for (int n = 0; n < ny; ++n) {
        DU(ICY + n, i) += dens * W(ICY + n, i);
        DU(IDN, i) -= dens * W(ICY + n, i);
      }
      DU(IVX + dir, i) = delta[i](1);
      DU(IVX + (IVY - IVX + dir) % 3, i) = delta[i](2);
      DU(IVX + (IVZ - IVX + dir) % 3, i) = delta[i](3);
      DU(IPR, i) = delta[i](4);
    }
  }

  // if (!first_block) SendBuffer(delta[il], bblock);
}

}  // namespace snap

#undef DU
#undef W
