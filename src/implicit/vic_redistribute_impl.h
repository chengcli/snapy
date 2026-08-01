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

  // reduction == 1: B_dry. (The per-species reductions that used to live at
  // reduction >= 2 were computed but never consumed -- their structural term
  // was disabled in #155 -- and are gone: species conservation is now handled
  // in flux form by vic_species_column below. Removing them also removes a
  // latent scratch overflow for ny > N*N - 2.)
  T dryfrac = 1;
  for (int n = 0; n < ny; ++n) dryfrac -= W(ICY + n, i);
  return residual_volume * dryfrac;
}

template <typename T, int N>
void DISPATCH_MACRO vic_column_reduce(T* du, T* w,
                                      Eigen::Matrix<T, N, 1>* delta, T* vol,
                                      T* scratch, int il, int iu, int ny,
                                      int stride1, int stride2) {
  for (int reduction = 0; reduction < 2; ++reduction) {
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

// Component B: implicit species transport in FLUX form (serial per column).
//
// The MS-VIC solve produces a per-cell total-mass correction
//   diffusion_fix_i = delta_i(0) - explicit_total_i          [density/step].
// The legacy species update applied it pointwise, pro-rata by the cell's own
// mixing ratio (du += diffusion_fix * y_i). That form (a) does not conserve
// species mass over the column whenever y varies with height, and (b) cannot
// advect a composition contrast (arriving mass always takes the receiving
// cell's own y). This pass converts the correction into the unique implied
// face mass transfer instead:
//
//   phi_i    = diffusion_fix_i * V_i                          [mass/step]
//   phi'_i   = phi_i - R * m_i / sum(m)   (R = sum(phi): the column residual
//              with no flux representation; the dry structural term repairs
//              exactly this part for IDN, so species see only the
//              flux-representable component)
//   M_{i+1/2} = M_{i-1/2} - phi'_i,  M_{-1/2} = 0  (closed bottom)
//              => M at the top face telescopes to exactly zero.
//
// Species then move by M * y_donor per face (donor-upwinded, compositionally
// active), with exact sequential availability clamping: a face never moves
// more than the donor holds after the explicit update (which is non-negative
// under dynamics: positivity). Conservation is exact by construction: every
// face's transfer is added to one cell and subtracted from its neighbour.
//
// Writes ONLY the mass_fix buffer (never du), so vic_redistribute_cell stays
// cell-parallel and its diffusion_fix recomputation sees pristine du:
//   MASS(IDN, i)      : scratch during the pass (overwritten with
//                       diffusion_fix by vic_redistribute_cell afterwards)
//   MASS(IVX+dir, i)  : M through the face BELOW cell i  [mass/step]
//                       (consumed by the passive-scalar update in meshblock)
//   MASS(ICY+n, i)    : the species increment [density/step] that
//                       vic_redistribute_cell applies when species_flux is on
// Assumes the whole column lives on this rank (implicit is nb1 = 1 by
// decision; no distributed-column support planned).
template <typename T, int N>
void DISPATCH_MACRO vic_species_column(T* du, T* w, T* mass_fix,
                                       Eigen::Matrix<T, N, 1>* delta, T* vol,
                                       int nlayer, int dir, int ny,
                                       int stride1, int stride2) {
  // pass 1: per-cell mass increments and the column residual
  T R = 0, S = 0;
  for (int i = 0; i < nlayer; ++i) {
    T explicit_total = DU(IDN, i);
    for (int n = 0; n < ny; ++n) explicit_total += DU(ICY + n, i);
    T phi = (delta[i](0) - explicit_total) * VOL(i);
    MASS(IDN, i) = phi;
    R += phi;
    S += W(IDN, i) * VOL(i);
  }

  // pass 2: face transfers by prefix sum of the zero-sum part
  T M = 0;
  for (int i = 0; i < nlayer; ++i) {
    MASS(IVX + dir, i) = M;
    M -= MASS(IDN, i) - R * (W(IDN, i) * VOL(i) / S);
  }
  // (M here is the top-face transfer: exactly zero up to roundoff)

  // pass 3: donor-upwinded species transfer with sequential availability
  // clamping; carry the running availability of the lower cell upward
  for (int n = 0; n < ny; ++n) {
    T avail = (W(IDN, 0) * W(ICY + n, 0) + DU(ICY + n, 0)) * VOL(0);
    if (avail < 0) avail = 0;
    for (int i = 0; i + 1 < nlayer; ++i) {
      T Mf = MASS(IVX + dir, i + 1);  // face between cells i and i+1
      T avail_up =
          (W(IDN, i + 1) * W(ICY + n, i + 1) + DU(ICY + n, i + 1)) * VOL(i + 1);
      if (avail_up < 0) avail_up = 0;

      T q = Mf > 0 ? Mf * W(ICY + n, i) : Mf * W(ICY + n, i + 1);
      if (q > avail) q = avail;          // draining cell i upward
      if (q < -avail_up) q = -avail_up;  // draining cell i+1 downward

      MASS(ICY + n, i) -= q / VOL(i);
      MASS(ICY + n, i + 1) += q / VOL(i + 1);
      avail = avail_up + q;
    }
  }
}

// Per-cell MS-VIC redistribution map. Reads delta, W, VOL and the per-column
// reduction scalars from `scratch`, writes the final tendencies DU for cell i.
template <typename T, int N>
void DISPATCH_MACRO vic_redistribute_cell(T* du, T* w, T* mass_fix,
                                          Eigen::Matrix<T, N, 1>* delta, T* vol,
                                          T const* scratch, int i, int dir,
                                          int ny, int nvapor, int stride1,
                                          int stride2, int species_flux = 0) {
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

  if (species_flux) {
    // Component B: apply the conservative donor-upwinded face transfers
    // precomputed by vic_species_column (stored in mass_fix).
    for (int n = 0; n < ny; ++n) {
      DU(ICY + n, i) = DU(ICY + n, i) + MASS(ICY + n, i);
    }
  } else {
    // Legacy pointwise pro-rata update: not conservative over the column
    // when y varies with height, and compositionally inert.
    for (int n = 0; n < ny; ++n) {
      DU(ICY + n, i) = DU(ICY + n, i) + diffusion_fix * W(ICY + n, i);
    }
  }
}

}  // namespace snap

#undef DU
#undef W
#undef MASS
#undef VOL
