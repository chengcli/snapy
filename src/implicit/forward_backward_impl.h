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
#define VOL(n) vol[(n) * stride2]

namespace snap {

template <typename T>
inline DISPATCH_MACRO T capped_exp(T x) {
  // return x >= 0 ? 2. - std::exp(-x) : std::exp(x);
  return 1. + 2. / M_PI * atan(M_PI / 2. * x);
}

template <typename T, int N>
void DISPATCH_MACRO ForwardSweep(Eigen::Matrix<T, N, N>* a,
                                 Eigen::Matrix<T, N, N>* b,
                                 Eigen::Matrix<T, N, N>* c,
                                 Eigen::Matrix<T, N, 1>* delta, T* du,
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
  Eigen::Matrix<T, N, N, Eigen::RowMajor> A;
  Eigen::Matrix<T, N, N + 1, Eigen::RowMajor> solved;

  // if (!first_block) {
  // RecvBuffer(a[il - 1], delta[il - 1], bblock);
  // a[il] = (a[il] - b[il] * a[il - 1]).inverse().eval();
  // delta[il] = a[il] * (rhs - b[il] * delta[il - 1]);
  // a[il] *= c[il];
  //} else {
  if constexpr (N > 4) {
    A = a[il];
    for (int n = 0; n < N; ++n) indx[n] = n;
    ludcmp(A, indx);
    solved.template leftCols<N>() = c[il];
    solved.col(N) = rhs;
    lubksb(A, indx, solved);
    a[il] = solved.template leftCols<N>();
    delta[il] = solved.col(N);
  } else {  // small matrix
    a[il] = a[il].inverse().eval();
    delta[il] = a[il] * rhs;
    a[il] = a[il] * c[il];
  }
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
      A = a[i];
      for (int n = 0; n < N; ++n) indx[n] = n;
      ludcmp(A, indx);
      solved.template leftCols<N>() = c[i];
      solved.col(N) = rhs - b[i] * delta[i - 1];
      lubksb(A, indx, solved);
      a[i] = solved.template leftCols<N>();
      delta[i] = solved.col(N);
    } else {  // small matrix
      a[i] = a[i].inverse().eval();
      delta[i] = a[i] * (rhs - b[i] * delta[i - 1]);
      a[i] = a[i] * c[i];
    }
  }

  // SaveCoefficients(a, delta, il, iu);
  // if (!last_block) SendBuffer(a[iu], delta[iu], tblock);
}

template <typename T, int N>
void DISPATCH_MACRO BackwardSubstitution(T* du, T* w, Eigen::Matrix<T, N, N>* a,
                                         Eigen::Matrix<T, N, 1>* delta, T* vol,
                                         int il, int iu, int dir, int ny,
                                         int stride1, int stride2,
                                         bool first_block, bool last_block) {
  // LoadCoefficients(a, delta, il, iu);
  // if (!last_block) {
  //   RecvBuffer(delta[iu + 1], tblock);
  //   delta[iu] -= a[iu] * delta[iu + 1];
  // }

  // update solutions, i=iu
  for (int i = iu - 1; i >= il; --i) delta[i] -= a[i] * delta[i + 1];

  T denom = 0;
  T dry_struct = 0;

  /// DU starts as the explicit constituent tendencies. delta[i](0) is the
  /// implicit total-density tendency after the VIC solve.
  /// The tridiagnonal coefficients are no longer needed after back substitution
  /// Use them as scratch space for temporary storage:
  /// - a[i](0) -> (Delta rho'_i * vol_i)
  /// - a[i](1) -> dry mass fraction
  /// - a[i](2) -> dry mass fraction correction factor

  for (int i = il; i <= iu; ++i) {
    a[i](0) = delta[i](0) * VOL(i);
    denom += a[i](0) * a[i](0);

    a[i](1) = 1;
    for (int n = 0; n < ny; ++n) a[i](1) -= W(ICY + n, i);
    dry_struct += (DU(IDN, i) - delta[i](0) * a[i](1)) * VOL(i);
  }

  /// MS-VIC redistribution:
  ///   y'_ij = y_ij + W_i S_j / sum_k W_k^2,
  /// with W_i = Delta rho'_i V_i and
  /// S_j = sum_i (Delta rho_i - Delta rho'_i) V_i y_ij.

  for (int i = il; i <= iu; ++i) {
    a[i](2) = capped_exp(a[i](0) * dry_struct / (denom * a[i](1)));
    DU(IDN, i) = delta[i](0) * a[i](1) * a[i](2);

    if constexpr (N == 3) {  // partial matrix
      DU(IVX + dir, i) = delta[i](1);
      DU(IPR, i) = delta[i](2);
    } else {  // full matrix
      DU(IVX + dir, i) = delta[i](1);
      DU(IVX + (IVY - IVX + dir) % 3, i) = delta[i](2);
      DU(IVX + (IVZ - IVX + dir) % 3, i) = delta[i](3);
      DU(IPR, i) = delta[i](4);
    }
  }

  for (int n = 0; n < ny; ++n) {
    T species_struct = 0;
    for (int i = il; i <= iu; ++i) {
      species_struct += (DU(ICY + n, i) - delta[i](0) * W(ICY + n, i)) * VOL(i);
    }

    for (int i = il; i <= iu; ++i) {
      T corr = capped_exp(a[i](0) * species_struct / (denom * W(ICY + n, i)));
      DU(ICY + n, i) = delta[i](0) * W(ICY + n, i) * corr;
      if (3 + n < N * N) a[i](3 + n) = corr;
    }
  }

  // if (!first_block) SendBuffer(delta[il], bblock);
}

}  // namespace snap

#undef VOL
#undef DU
#undef W
