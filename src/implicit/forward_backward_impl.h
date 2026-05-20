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

  T vol_sum = 0;
  T dry_column_error = 0;

  /// Update conserved variables. DU starts as the original explicit
  /// tendency, while delta[i](0) is the implicit adjusted total-density
  /// tendency. The total-density correction
  ///
  ///   c_i = delta_i(0) - (du_IDN,i + sum_n du_ICY+n,i)
  ///
  /// is first redistributed in each cell by the original mass fractions in W:
  /// f_ICY+n,i = W(ICY+n,i), f_dry,i = 1 - sum_n W(ICY+n,i). This preserves the
  /// implicit total-density tendency per cell, but it changes the constituent
  /// column integrals by D_k = sum_i V_i f_k,i c_i. Store c_i in delta[i](0)
  /// for the column-conservation pass below.
  for (int i = il; i <= iu; ++i) {
    T dens_corr = DU(IDN, i);
    for (int n = 0; n < ny; ++n) dens_corr += DU(ICY + n, i);
    dens_corr = delta[i](0) - dens_corr;
    delta[i](0) = dens_corr;

    T dry_frac = 1;
    for (int n = 0; n < ny; ++n) dry_frac -= W(ICY + n, i);

    T cell_vol = vol[i * stride2];
    vol_sum += cell_vol;
    dry_column_error += cell_vol * dry_frac * dens_corr;

    if constexpr (N == 3) {  // partial matrix
      DU(IDN, i) += dens_corr * dry_frac;
      for (int n = 0; n < ny; ++n) {
        DU(ICY + n, i) += dens_corr * W(ICY + n, i);
      }
      DU(IVX + dir, i) = delta[i](1);
      DU(IPR, i) = delta[i](2);
    } else {  // full matrix
      DU(IDN, i) += dens_corr * dry_frac;
      for (int n = 0; n < ny; ++n) {
        DU(ICY + n, i) += dens_corr * W(ICY + n, i);
      }
      DU(IVX + dir, i) = delta[i](1);
      DU(IVX + (IVY - IVX + dir) % 3, i) = delta[i](2);
      DU(IVX + (IVZ - IVX + dir) % 3, i) = delta[i](3);
      DU(IPR, i) = delta[i](4);
    }
  }

  /// Restore each constituent's original volume-weighted column tendency by
  /// adding a uniform offset a_k to every cell:
  ///
  ///   a_k = -D_k / sum_i V_i,  D_k = sum_i V_i f_k,i c_i.
  ///
  /// Applying the species one at a time avoids dynamic allocation in this
  /// CPU/CUDA templated routine. Dry air uses f_dry,i above.
  T dry_offset = -dry_column_error / vol_sum;
  for (int i = il; i <= iu; ++i) DU(IDN, i) += dry_offset;

  for (int n = 0; n < ny; ++n) {
    T species_column_error = 0;
    for (int i = il; i <= iu; ++i) {
      species_column_error += vol[i * stride2] * W(ICY + n, i) * delta[i](0);
    }

    T species_offset = -species_column_error / vol_sum;
    for (int i = il; i <= iu; ++i) DU(ICY + n, i) += species_offset;
  }

  // if (!first_block) SendBuffer(delta[il], bblock);
}

}  // namespace snap

#undef DU
#undef W
