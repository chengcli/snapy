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
// (forward_sweep_impl.h). Backward substitution and the donor-upwinded
// constituent transport both retain their required serial layer dependency;
// the final per-cell update is exposed separately so CUDA can parallelize it.
template <typename T, int N>
void DISPATCH_MACRO vic_backward_substitute(Eigen::Matrix<T, N, N>* a,
                                            Eigen::Matrix<T, N, 1>* delta,
                                            int il, int iu) {
  for (int i = iu - 1; i >= il; --i) delta[i] -= a[i] * delta[i + 1];
}

// Component B: implicit constituent transport in FLUX form (serial per
// column).
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
//              with no closed-face flux representation, removed in proportion
//              to cell mass so every constituent sees only the
//              flux-representable component)
//   M_{i+1/2} = M_{i-1/2} - phi'_i,  M_{-1/2} = 0  (closed bottom)
//              => M at the top face telescopes to exactly zero.
//
// Dry gas and every species then move by M * y_donor per face
// (donor-upwinded, compositionally active), with exact sequential availability
// clamping: a face never moves more than the donor holds after the explicit
// update. Conservation is exact by construction: every face's transfer is
// added to one cell and subtracted from its neighbour.
//
// Writes ONLY the mass_fix buffer (never du), so vic_redistribute_cell stays
// cell-parallel:
//   MASS(IDN, i)      : dry-gas increment [density/step]
//   MASS(IVX+dir, i)  : M through the face BELOW cell i  [mass/step]
//                       (consumed by the passive-scalar update in meshblock)
//   MASS(ICY+n, i)    : the species increment [density/step] that
//                       vic_redistribute_cell applies
//   MASS(IPR, i)      : temporary phi storage, cleared before returning
// Assumes the whole column lives on this rank (implicit is nb1 = 1 by
// decision; no distributed-column support planned).
template <typename T, int N>
void DISPATCH_MACRO vic_constituent_column(T* du, T* w, T* mass_fix,
                                           Eigen::Matrix<T, N, 1>* delta,
                                           T* vol, int nlayer, int dir, int ny,
                                           int stride1, int stride2) {
  // pass 1: per-cell mass increments and the column residual
  T R = 0, S = 0;
  for (int i = 0; i < nlayer; ++i) {
    T explicit_total = DU(IDN, i);
    for (int n = 0; n < ny; ++n) explicit_total += DU(ICY + n, i);
    T phi = (delta[i](0) - explicit_total) * VOL(i);
    MASS(IDN, i) = 0;
    for (int n = 0; n < ny; ++n) MASS(ICY + n, i) = 0;
    MASS(IPR, i) = phi;
    R += phi;
    S += W(IDN, i) * VOL(i);
  }

  // pass 2: face transfers by prefix sum of the zero-sum part
  T M = 0;
  for (int i = 0; i < nlayer; ++i) {
    MASS(IVX + dir, i) = M;
    M -= MASS(IPR, i) - R * (W(IDN, i) * VOL(i) / S);
    MASS(IPR, i) = 0;
  }
  // (M here is the top-face transfer: exactly zero up to roundoff)

  // pass 3a: donor-upwinded dry-gas transfer with sequential availability
  T dryfrac = 1;
  for (int n = 0; n < ny; ++n) dryfrac -= W(ICY + n, 0);
  T avail = (W(IDN, 0) * dryfrac + DU(IDN, 0)) * VOL(0);
  if (avail < 0) avail = 0;
  for (int i = 0; i + 1 < nlayer; ++i) {
    T dryfrac_up = 1;
    for (int n = 0; n < ny; ++n) dryfrac_up -= W(ICY + n, i + 1);
    T avail_up = (W(IDN, i + 1) * dryfrac_up + DU(IDN, i + 1)) * VOL(i + 1);
    if (avail_up < 0) avail_up = 0;

    T Mf = MASS(IVX + dir, i + 1);
    T q = Mf > 0 ? Mf * dryfrac : Mf * dryfrac_up;
    if (q > avail) q = avail;
    if (q < -avail_up) q = -avail_up;

    MASS(IDN, i) -= q / VOL(i);
    MASS(IDN, i + 1) += q / VOL(i + 1);
    avail = avail_up + q;
    dryfrac = dryfrac_up;
  }

  // pass 3b: donor-upwinded species transfer with sequential availability
  // clamping; carry the running availability of the lower cell upward
  for (int n = 0; n < ny; ++n) {
    avail = (W(IDN, 0) * W(ICY + n, 0) + DU(ICY + n, 0)) * VOL(0);
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

// Per-cell MS-VIC redistribution map. Reads delta and the precomputed
// constituent increments from mass_fix, then writes the final tendencies DU.
template <typename T, int N>
void DISPATCH_MACRO vic_redistribute_cell(T* du, T* mass_fix,
                                          Eigen::Matrix<T, N, 1>* delta, int i,
                                          int dir, int ny, int stride1,
                                          int stride2) {
  DU(IDN, i) += MASS(IDN, i);

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
    DU(ICY + n, i) += MASS(ICY + n, i);
  }
}

}  // namespace snap

#undef DU
#undef W
#undef MASS
#undef VOL
