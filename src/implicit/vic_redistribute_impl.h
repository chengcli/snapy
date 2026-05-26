#pragma once

// eigen
#include <Eigen/Dense>

// base
#include <configure.h>

// snap
#include "forward_backward_impl.h"

#define DU(n, i) du[(n) * stride1 + (i) * stride2]
#define W(n, i) w[(n) * stride1 + (i) * stride2]
#define VOL(n) vol[(n) * stride2]

namespace snap {

// Split form of the MS-VIC BackwardSubstitution (forward_backward_impl.h) for
// the GPU. The serial work (backward substitution + the per-column reduction
// sums) stays per-column here; the per-cell redistribution map moves to
// vic_redistribute_cell() so it can run as a cell-parallel kernel.
//
// `scratch` is the column's reduction storage. scratch[0]=sum_k A_k^2,
// scratch[1]=B_dry, scratch[2+n]=B_species[n]. DU is NOT written here.
template <typename T, int N>
void DISPATCH_MACRO vic_backward_reduce(T* du, T* w, Eigen::Matrix<T, N, N>* a,
                                        Eigen::Matrix<T, N, 1>* delta, T* vol,
                                        T* scratch, int il, int iu, int dir,
                                        int ny, int stride1, int stride2,
                                        bool conservation) {
  // backward substitution
  for (int i = iu - 1; i >= il; --i) delta[i] -= a[i] * delta[i + 1];

  // Per-column reduction sums for the exact MS-VIC formula.
  T sum_a2 = 0;
  T b_dry = 0;
  if (!conservation) {
    scratch[0] = 0;
    scratch[1] = 0;
    for (int n = 0; n < ny; ++n) scratch[2 + n] = 0;
    return;
  }

  for (int i = il; i <= iu; ++i) {
    T explicit_total = DU(IDN, i);
    T dryfrac = 1;
    for (int n = 0; n < ny; ++n) {
      explicit_total += DU(ICY + n, i);
      dryfrac -= W(ICY + n, i);
    }

    T a_i = W(IDN, i) * VOL(i);
    sum_a2 += a_i * a_i;
    b_dry += (explicit_total - delta[i](0)) * dryfrac * VOL(i);
  }
  scratch[0] = sum_a2;
  scratch[1] = b_dry;

  for (int n = 0; n < ny; ++n) {
    T b_species = 0;
    for (int i = il; i <= iu; ++i) {
      T explicit_total = DU(IDN, i);
      for (int m = 0; m < ny; ++m) explicit_total += DU(ICY + m, i);
      b_species += (explicit_total - delta[i](0)) * W(ICY + n, i) * VOL(i);
    }
    scratch[2 + n] = b_species;
  }
}

// Per-cell MS-VIC redistribution map. Reads delta, W, VOL and the per-column
// reduction scalars from `scratch`, writes the final tendencies DU for cell i.
template <typename T, int N>
void DISPATCH_MACRO vic_redistribute_cell(T* du, T* w,
                                          Eigen::Matrix<T, N, 1>* delta, T* vol,
                                          T const* scratch, int i, int dir,
                                          int ny, int stride1, int stride2,
                                          bool conservation) {
  T sum_a2 = scratch[0];

  T a_i = W(IDN, i) * VOL(i);
  T dryfrac = 1;
  for (int n = 0; n < ny; ++n) dryfrac -= W(ICY + n, i);

  T explicit_total = DU(IDN, i);
  for (int n = 0; n < ny; ++n) explicit_total += DU(ICY + n, i);

  T dry_structural = conservation && sum_a2 > 0
                         ? W(IDN, i) * a_i * scratch[1] / sum_a2
                         : static_cast<T>(0);
  DU(IDN, i) =
      DU(IDN, i) + (delta[i](0) - explicit_total) * dryfrac + dry_structural;

  if constexpr (N == 3) {  // partial matrix
    DU(IVX + dir, i) = delta[i](1);
    DU(IPR, i) = delta[i](2);
  } else {  // full matrix
    DU(IVX + dir, i) = delta[i](1);
    DU(IVX + (IVY - IVX + dir) % 3, i) = delta[i](2);
    DU(IVX + (IVZ - IVX + dir) % 3, i) = delta[i](3);
    DU(IPR, i) = delta[i](4);
  }

  for (int n = 0; n < ny; ++n) {
    T structural = conservation && sum_a2 > 0
                       ? W(IDN, i) * a_i * scratch[2 + n] / sum_a2
                       : static_cast<T>(0);
    DU(ICY + n, i) = DU(ICY + n, i) +
                     (delta[i](0) - explicit_total) * W(ICY + n, i) +
                     structural;
  }
}

}  // namespace snap

#undef DU
#undef W
#undef VOL
