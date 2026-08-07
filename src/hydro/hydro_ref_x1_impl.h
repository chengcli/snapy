#pragma once

// base
#include <configure.h>

// C/C++
#include <cmath>
#include <limits>

// snap
#include <snap/snap.h>

namespace snap {

template <typename T>
inline DISPATCH_MACRO T hydro_ref_x1_face(T const* psf_lo, T const* psf_hi,
                                          int flat, int face, int nc1) {
  if (face < nc1) return psf_lo[flat + face];
  return psf_hi[flat + nc1 - 1];
}

template <typename T>
inline DISPATCH_MACRO void hydro_ref_x1_scan_impl(
    T const* w, T const* dx1f, T const* anchor, T const* gam, T const* kbot_in,
    T* psf_lo, T* psf_hi, int column, int ncolumns, int nc1, int is, int iu,
    T grav, T* kbot, T* inv_gamma) {
  int ncells = ncolumns * nc1;
  int flat = column * nc1;
  T top_anchor;
  if (anchor) {
    top_anchor = anchor[column];
  } else {
    T rho_top = w[IDN * ncells + flat + iu];
    T pres_top = w[IPR * ncells + flat + iu];
    T rdt_top = pres_top / rho_top;
    top_anchor = pres_top * exp(-grav * T(0.5) * dx1f[iu] / rdt_top);
  }

  T face = top_anchor;
  T min_positive = std::numeric_limits<T>::min();
  for (int i = iu; i >= 0; --i) {
    T dp = grav * w[IDN * ncells + flat + i] * dx1f[i];
    T lo = face + dp;
    T hi = face;
    psf_lo[flat + i] = lo > min_positive ? lo : min_positive;
    psf_hi[flat + i] = hi > min_positive ? hi : min_positive;
    face = lo;
  }

  face = top_anchor;
  for (int i = iu + 1; i < nc1; ++i) {
    T dp = grav * w[IDN * ncells + flat + i] * dx1f[i];
    T lo = face;
    T hi = lo - dp;
    psf_lo[flat + i] = lo > min_positive ? lo : min_positive;
    psf_hi[flat + i] = hi > min_positive ? hi : min_positive;
    face = hi;
  }

  T gamma = gam[column];
  // kbot_in (relayed from the block owning the PHYSICAL bottom, hydro.cpp)
  // makes the density reference a SINGLE global isentrope across an x1 (nb1>1)
  // decomposition. The block-local fallback is the nb1=1 path, bit-unchanged.
  // A per-block kbot made adjacent blocks decompose rho against different
  // isentropes, so the two sides of every x1 seam reconstructed different
  // face states -- one of the two defects behind the decomposition-dependent
  // convective vigor.
  if (kbot_in) {
    *kbot = kbot_in[column];
  } else {
    T rho_bot = w[IDN * ncells + flat + is];
    T pres_bot = w[IPR * ncells + flat + is];
    *kbot = pres_bot / pow(rho_bot, gamma);
  }
  *inv_gamma = T(1) / gamma;
}

template <typename T>
inline DISPATCH_MACRO void hydro_ref_x1_cell_impl(
    T const* w, T const* dx1f, T const* psf_lo, T const* psf_hi, T* pref,
    T* dsf, T* dref, int column, int i, int ncolumns, int nc1, T grav,
    bool uniform, bool phys_in, bool phys_out, T kbot, T inv_gamma) {
  int ncells = ncolumns * nc1;
  int flat = column * nc1;
  T lo = psf_lo[flat + i];
  T hi = psf_hi[flat + i];
  T cell_pref = T(0.5) * (lo + hi);

  if (uniform) {
    constexpr double w6[6] = {11. / 1440., -31. / 480., 401. / 720.,
                              401. / 720., -31. / 480., 11. / 1440.};
    if (i >= 2 && i < nc1 - 2) {
      T six = T(0);
      for (int m = 0; m < 6; ++m) {
        six +=
            T(w6[m]) * hydro_ref_x1_face(psf_lo, psf_hi, flat, i - 2 + m, nc1);
      }
      T lower = lo < hi ? lo : hi;
      T upper = lo > hi ? lo : hi;
      if (six >= lower && six <= upper) cell_pref = six;
    }

    constexpr double w6e[2][6] = {
        {95. / 288., 1427. / 1440., -133. / 240., 241. / 720., -173. / 1440.,
         3. / 160.},
        {-3. / 160., 637. / 1440., 511. / 720., -43. / 240., 77. / 1440.,
         -11. / 1440.},
    };
    if (!phys_in && i < 2) {
      T val = T(0);
      for (int m = 0; m < 6; ++m) {
        val += T(w6e[i][m]) * hydro_ref_x1_face(psf_lo, psf_hi, flat, m, nc1);
      }
      T lower = lo < hi ? lo : hi;
      T upper = lo > hi ? lo : hi;
      if (val >= lower && val <= upper) cell_pref = val;
    }
    if (!phys_out && i >= nc1 - 2) {
      int sigma = i - (nc1 - 5);
      int row = 4 - sigma;
      T val = T(0);
      for (int m = 0; m < 6; ++m) {
        val += T(w6e[row][5 - m]) *
               hydro_ref_x1_face(psf_lo, psf_hi, flat, nc1 - 5 + m, nc1);
      }
      T lower = lo < hi ? lo : hi;
      T upper = lo > hi ? lo : hi;
      if (val >= lower && val <= upper) cell_pref = val;
    }
  } else {
    T dp = grav * w[IDN * ncells + flat + i] * dx1f[i];
    T ratio = lo / hi;
    cell_pref =
        fabs(ratio - T(1)) < T(1.e-6) ? T(0.5) * (lo + hi) : dp / log(ratio);
  }

  pref[flat + i] = cell_pref;
  dref[flat + i] = pow(cell_pref / kbot, inv_gamma);
  dsf[flat + i] = pow(lo / kbot, inv_gamma);
}

template <typename T>
inline DISPATCH_MACRO void hydro_ref_x1_impl(
    T const* w, T const* dx1f, T const* anchor, T const* gam, T const* kbot_in,
    T* psf_lo, T* psf_hi, T* pref, T* dsf, T* dref, int column, int ncolumns,
    int nc1, int is, int iu, T grav, bool uniform, bool phys_in,
    bool phys_out) {
  T kbot;
  T inv_gamma;
  hydro_ref_x1_scan_impl(w, dx1f, anchor, gam, kbot_in, psf_lo, psf_hi, column,
                         ncolumns, nc1, is, iu, grav, &kbot, &inv_gamma);
  for (int i = 0; i < nc1; ++i) {
    hydro_ref_x1_cell_impl(w, dx1f, psf_lo, psf_hi, pref, dsf, dref, column, i,
                           ncolumns, nc1, grav, uniform, phys_in, phys_out,
                           kbot, inv_gamma);
  }
}

}  // namespace snap
