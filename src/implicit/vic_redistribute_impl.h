#pragma once

// C/C++
#include <algorithm>
#include <cmath>

// eigen
#include <Eigen/Dense>

// base
#include <configure.h>

// snap
#include "forward_backward_impl.h"  // capped_exp + index constants

#define DU(n, i) du[(n) * stride1 + (i) * stride2]
#define W(n, i) w[(n) * stride1 + (i) * stride2]
#define VOL(n) vol[(n) * stride2]

namespace snap {

// Split form of the MS-VIC BackwardSubstitution (forward_backward_impl.h) for
// the GPU. The serial work (backward substitution + the per-column reduction
// sums) stays per-column here; the per-cell capped_exp redistribution map moves
// to vic_redistribute_cell() so it can run as a cell-parallel kernel. Together
// these reproduce BackwardSubstitution's arithmetic (modulo FMA contraction).
//
// `scratch` is the column's reduction storage (the b off-diagonal buffer, free
// after ForwardSweep): scratch[0]=denom, scratch[1]=dry_struct,
// scratch[2+n]=species_struct[n]. DU is NOT written here.
template <typename T, int N>
void DISPATCH_MACRO vic_backward_reduce(T* du, T* w, Eigen::Matrix<T, N, N>* a,
                                        Eigen::Matrix<T, N, 1>* delta, T* vol,
                                        T* scratch, int il, int iu, int dir,
                                        int ny, int stride1, int stride2) {
  // backward substitution
  for (int i = iu - 1; i >= il; --i) delta[i] -= a[i] * delta[i + 1];

  // per-column reduction sums (read the original, un-redistributed DU)
  T denom = 0;
  T dry_struct = 0;
  for (int i = il; i <= iu; ++i) {
    T a0 = delta[i](0) * VOL(i);
    denom += a0 * a0;
    T dryfrac = 1;
    for (int n = 0; n < ny; ++n) dryfrac -= W(ICY + n, i);
    dry_struct += (DU(IDN, i) - delta[i](0) * dryfrac) * VOL(i);
  }
  scratch[0] = denom;
  scratch[1] = dry_struct;

  for (int n = 0; n < ny; ++n) {
    T species_struct = 0;
    for (int i = il; i <= iu; ++i)
      species_struct += (DU(ICY + n, i) - delta[i](0) * W(ICY + n, i)) * VOL(i);
    scratch[2 + n] = species_struct;
  }
}

// Per-cell MS-VIC redistribution map. Reads delta, W, VOL and the per-column
// reduction scalars from `scratch`, writes the final tendencies DU for cell i.
template <typename T, int N>
void DISPATCH_MACRO vic_redistribute_cell(T* du, T* w,
                                          Eigen::Matrix<T, N, 1>* delta, T* vol,
                                          T const* scratch, int i, int dir,
                                          int ny, int stride1, int stride2) {
  T denom = scratch[0];
  T dry_struct = scratch[1];

  T a0 = delta[i](0) * VOL(i);
  T dryfrac = 1;
  for (int n = 0; n < ny; ++n) dryfrac -= W(ICY + n, i);

  DU(IDN, i) =
      delta[i](0) * dryfrac *
      capped_exp(a0 * dry_struct / std::max((T)1.e-12, denom * dryfrac));

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
    T corr = capped_exp(a0 * scratch[2 + n] /
                        std::max((T)1.e-12, denom * W(ICY + n, i)));
    DU(ICY + n, i) = delta[i](0) * W(ICY + n, i) * corr;
  }
}

}  // namespace snap

#undef DU
#undef W
#undef VOL
