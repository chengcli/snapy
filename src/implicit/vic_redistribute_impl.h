#pragma once

// eigen
#include <Eigen/Dense>

// base
#include <configure.h>

// snap
#include "forward_sweep_impl.h"

#define DU(n, i) du[(n) * stride1 + (i) * stride2]
#define W(n, i) w[(n) * stride1 + (i) * stride2]
#define MASS(n, i) mass_fix[(n) * stride1 + (i) * stride2]
#define VOL(n) vol[(n) * stride2]

namespace snap {

// Split form of the MS-VIC redistribution around ForwardSweep
// (forward_sweep_impl.h). Backward substitution retains its required serial
// layer dependency. The reduction terms and per-cell redistribution map are
// exposed separately so CUDA can parallelize them.
//
// `scratch` is the column's reduction storage. scratch[0]=sum_k A_k^2,
// scratch[1]=B_dry, scratch[2+n]=B_species[n]. DU is NOT written here.
template <typename T, int N>
void DISPATCH_MACRO vic_backward_substitute(Eigen::Matrix<T, N, N>* a,
                                            Eigen::Matrix<T, N, 1>* delta,
                                            int il, int iu) {
  for (int i = iu - 1; i >= il; --i) delta[i] -= a[i] * delta[i + 1];
}

// Return cell i's contribution to one reduction quantity. reduction=0 is
// sum(A_i^2), reduction=1 is B_dry, and reduction=2+n is B_species[n].
template <typename T, int N>
T DISPATCH_MACRO vic_reduction_term(T* du, T* w, Eigen::Matrix<T, N, 1>* delta,
                                    T* vol, int i, int reduction, int ny,
                                    int stride1, int stride2) {
  if (reduction == 0) {
    T a_i = W(IDN, i) * VOL(i);
    return a_i * a_i;
  }

  T explicit_total = DU(IDN, i);
  for (int n = 0; n < ny; ++n) explicit_total += DU(ICY + n, i);
  T residual_volume = (explicit_total - delta[i](0)) * VOL(i);

  if (reduction == 1) {
    T dryfrac = 1;
    for (int n = 0; n < ny; ++n) dryfrac -= W(ICY + n, i);
    return residual_volume * dryfrac;
  }

  return residual_volume * W(ICY + reduction - 2, i);
}

template <typename T, int N>
void DISPATCH_MACRO vic_column_reduce(T* du, T* w,
                                      Eigen::Matrix<T, N, 1>* delta, T* vol,
                                      T* scratch, int il, int iu, int ny,
                                      int stride1, int stride2) {
  for (int reduction = 0; reduction < 2 + ny; ++reduction) {
    T sum = 0;
    for (int i = il; i <= iu; ++i)
      sum += vic_reduction_term(du, w, delta, vol, i, reduction, ny, stride1,
                                stride2);
    scratch[reduction] = sum;
  }
}

// Serial convenience wrapper used by the CPU path and unit tests.
template <typename T, int N>
void DISPATCH_MACRO vic_backward_reduce(T* du, T* w, Eigen::Matrix<T, N, N>* a,
                                        Eigen::Matrix<T, N, 1>* delta, T* vol,
                                        T* scratch, int il, int iu, int dir,
                                        int ny, int stride1, int stride2) {
  (void)dir;
  vic_backward_substitute(a, delta, il, iu);
  vic_column_reduce(du, w, delta, vol, scratch, il, iu, ny, stride1, stride2);
}

// Per-cell MS-VIC redistribution map. Reads delta, W, VOL and the per-column
// reduction scalars from `scratch`, writes the final tendencies DU for cell i.
template <typename T, int N>
void DISPATCH_MACRO vic_redistribute_cell(T* du, T* w, T* mass_fix,
                                          Eigen::Matrix<T, N, 1>* delta, T* vol,
                                          T const* scratch, int i, int dir,
                                          int ny, int nvapor, int stride1,
                                          int stride2) {
  T sum_a2 = scratch[0];

  T a_i = W(IDN, i) * VOL(i);
  T dryfrac = 1;
  for (int n = 0; n < ny; ++n) dryfrac -= W(ICY + n, i);

  T explicit_total = DU(IDN, i);
  for (int n = 0; n < ny; ++n) explicit_total += DU(ICY + n, i);

  T diffusion_fix = delta[i](0) - explicit_total;
  MASS(IDN, i) = diffusion_fix;

  T dry_structural = W(IDN, i) * a_i * scratch[1] / sum_a2;
  DU(IDN, i) = DU(IDN, i) + diffusion_fix * dryfrac + dry_structural;

  if constexpr (N == 3) {  // partial matrix
    DU(IVX + dir, i) = delta[i](1);
    DU(IPR, i) = delta[i](2);
  } else {  // full matrix
    DU(IVX + dir, i) = delta[i](1);
    DU(IVX + (IVY - IVX + dir) % 3, i) = delta[i](2);
    DU(IVX + (IVZ - IVX + dir) % 3, i) = delta[i](3);
    DU(IPR, i) = delta[i](4);
  }

  /*for (int n = 0; n < nvapor; ++n) {
    T structural = W(IDN, i) * a_i * scratch[2 + n] / sum_a2;
    DU(ICY + n, i) =
        DU(ICY + n, i) + diffusion_fix * W(ICY + n, i) + structural;
  }*/

  for (int n = 0; n < ny; ++n) {
    DU(ICY + n, i) = DU(ICY + n, i) + diffusion_fix * W(ICY + n, i);
  }
}

}  // namespace snap

#undef DU
#undef W
#undef MASS
#undef VOL
