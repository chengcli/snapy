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
                                         bool first_block, bool last_block,
                                         bool conservation) {
  // LoadCoefficients(a, delta, il, iu);
  // if (!last_block) {
  //   RecvBuffer(delta[iu + 1], tblock);
  //   delta[iu] -= a[iu] * delta[iu + 1];
  // }

  // update solutions, i=iu
  for (int i = iu - 1; i >= il; --i) delta[i] -= a[i] * delta[i + 1];

  T sum_a2 = 0;

  /// DU starts as the explicit constituent tendencies. delta[i](0) is the
  /// implicit total-density tendency after the VIC solve.
  /// The tridiagnonal coefficients are no longer needed after back substitution
  /// Use them as scratch space for temporary storage:
  /// - a[i](0) -> explicit total density tendency, Delta rho_i
  /// - a[i](1) -> layer weight A_i = rho_i * V_i
  /// - a[i](2) -> dry mass fraction

  for (int i = il; i <= iu; ++i) {
    T explicit_total = DU(IDN, i);
    T dryfrac = 1;
    for (int n = 0; n < ny; ++n) {
      explicit_total += DU(ICY + n, i);
      dryfrac -= W(ICY + n, i);
    }

    a[i](0) = explicit_total;
    a[i](1) = W(IDN, i) * VOL(i);
    a[i](2) = dryfrac;
    sum_a2 += a[i](1) * a[i](1);
  }

  /// MS-VIC redistribution from ms_vic.tex:
  ///   Delta rho'_ij = Delta rho_ij + (Delta rho'_i - Delta rho_i) y_ij
  ///                    + rho_i A_i B_j / sum_k A_k^2,
  /// with A_i = rho_i V_i and
  ///      B_j = sum_i (Delta rho_i - Delta rho'_i) y_ij V_i.

  T b_dry = 0;
  for (int i = il; i <= iu; ++i) {
    b_dry += (a[i](0) - delta[i](0)) * a[i](2) * VOL(i);
  }

  for (int i = il; i <= iu; ++i) {
    T structural = conservation && sum_a2 > 0
                       ? W(IDN, i) * a[i](1) * b_dry / sum_a2
                       : static_cast<T>(0);
    DU(IDN, i) = DU(IDN, i) + (delta[i](0) - a[i](0)) * a[i](2) + structural;

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
    T b_species = 0;
    for (int i = il; i <= iu; ++i) {
      b_species += (a[i](0) - delta[i](0)) * W(ICY + n, i) * VOL(i);
    }

    for (int i = il; i <= iu; ++i) {
      T structural = conservation && sum_a2 > 0
                         ? W(IDN, i) * a[i](1) * b_species / sum_a2
                         : static_cast<T>(0);
      DU(ICY + n, i) =
          DU(ICY + n, i) + (delta[i](0) - a[i](0)) * W(ICY + n, i) + structural;
    }
  }

  // if (!first_block) SendBuffer(delta[il], bblock);
}

}  // namespace snap

#undef VOL
#undef DU
#undef W
