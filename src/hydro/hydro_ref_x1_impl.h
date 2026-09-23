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
inline DISPATCH_MACRO void hydro_ref_x1_scan_impl(T const* w, T const* dx1f,
                                                  T const* anchor, T* psf_lo,
                                                  T* psf_hi, int column,
                                                  int ncolumns, int nc1, int iu,
                                                  T grav) {
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
}

//! rho/p smoothed by a clamped 5-point binomial along x1: the reference must
//! track the column profile at LARGE scales only. An unsmoothed local rho/p
//! makes rho' degenerate with the pressure perturbation, so entropy/buoyancy
//! anomalies bypass the high-order reconstruction; a bottom-anchored
//! isentrope reference errs by orders of magnitude on a stratified column.
template <typename T>
inline DISPATCH_MACRO T hydro_ref_x1_rop_smooth(T const* w, int ncells,
                                                int flat, int nc1, int i,
                                                int jlo, int jhi) {
  T v[5];
  for (int m = -2; m <= 2; ++m) {
    int j = i + m;
    j = j < jlo ? jlo : (j > jhi ? jhi : j);
    v[m + 2] = w[IDN * ncells + flat + j] / w[IPR * ncells + flat + j];
  }
  return (v[0] + T(4) * v[1] + T(6) * v[2] + T(4) * v[3] + v[4]) / T(16);
}

template <typename T>
inline DISPATCH_MACRO void hydro_ref_x1_cell_impl(
    T const* w, T const* dx1f, T const* psf_lo, T const* psf_hi, T* pref,
    T* dsf, T* dref, int column, int i, int ncolumns, int nc1, int iu, T grav,
    bool uniform, bool phys_in, bool phys_out, bool wall_clamp) {
  int ncells = ncolumns * nc1;
  int flat = column * nc1;
  int il = nc1 - 1 - iu;
  // the six-face stencil never crosses a physical wall
  // A one-sided row spans SIX faces, so on a block with fewer than five x1
  // cells it reaches one face past the owned range -- at the OPPOSITE end from
  // the wall it is protecting. That is a seam face, i.e. the neighbour's own
  // data, and reading it is right; it is only unreadable when that end is a
  // physical wall too, which means the whole column is under five cells.
  bool thin = (iu - il + 1) < 5;
  // and the row has to FIT THE ARRAY. The inner row reads faces il..il+5, the
  // outer iu-4..iu+1, and with nc1 == il + iu + 1 both fit exactly when
  // iu >= 4. Below that the outer row's start goes negative -- and
  // hydro_ref_x1_face saturates above nc1 but has no lower guard, so it would
  // read before this column. nx1 >= 5 always satisfies it; a block thin enough
  // to fail it has fewer cells than it has ghosts.
  bool fits = iu >= 4;
  bool wall_in = wall_clamp && phys_in && i >= il && i < il + 2;
  bool wall_out = wall_clamp && phys_out && i > iu - 2 && i <= iu;
  bool clamp_in = wall_in && fits && !(thin && phys_out);
  bool clamp_out = wall_out && fits && !(thin && phys_in);
  T lo = psf_lo[flat + i];
  T hi = psf_hi[flat + i];
  T cell_pref = T(0.5) * (lo + hi);

  if (uniform) {
    constexpr double w6[6] = {11. / 1440., -31. / 480., 401. / 720.,
                              401. / 720., -31. / 480., 11. / 1440.};
    if (i >= 2 && i < nc1 - 2 && !wall_in && !wall_out) {
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
    if (clamp_in) {
      int sigma = i - il;
      T val = T(0);
      for (int m = 0; m < 6; ++m) {
        val += T(w6e[sigma][m]) *
               hydro_ref_x1_face(psf_lo, psf_hi, flat, il + m, nc1);
      }
      T lower = lo < hi ? lo : hi;
      T upper = lo > hi ? lo : hi;
      if (val >= lower && val <= upper) cell_pref = val;
    }
    if (clamp_out) {
      int s = iu + 1 - 5;
      int row = 4 - (i - s);
      T val = T(0);
      for (int m = 0; m < 6; ++m) {
        val += T(w6e[row][5 - m]) *
               hydro_ref_x1_face(psf_lo, psf_hi, flat, s + m, nc1);
      }
      T lower = lo < hi ? lo : hi;
      T upper = lo > hi ? lo : hi;
      if (val >= lower && val <= upper) cell_pref = val;
    }
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
  int jlo = (wall_clamp && phys_in) ? il : 0;
  int jhi = (wall_clamp && phys_out) ? iu : nc1 - 1;
  T rs = hydro_ref_x1_rop_smooth(w, ncells, flat, nc1, i, jlo, jhi);
  T rf = i > 0 ? T(0.5) * (hydro_ref_x1_rop_smooth(w, ncells, flat, nc1, i - 1,
                                                   jlo, jhi) +
                           rs)
               : rs;
  dref[flat + i] = cell_pref * rs;
  dsf[flat + i] = lo * rf;
}

template <typename T>
inline DISPATCH_MACRO void hydro_ref_x1_impl(
    T const* w, T const* dx1f, T const* anchor, T* psf_lo, T* psf_hi, T* pref,
    T* dsf, T* dref, int column, int ncolumns, int nc1, int iu, T grav,
    bool uniform, bool phys_in, bool phys_out, bool wall_clamp) {
  hydro_ref_x1_scan_impl(w, dx1f, anchor, psf_lo, psf_hi, column, ncolumns, nc1,
                         iu, grav);
  for (int i = 0; i < nc1; ++i) {
    hydro_ref_x1_cell_impl(w, dx1f, psf_lo, psf_hi, pref, dsf, dref, column, i,
                           ncolumns, nc1, iu, grav, uniform, phys_in, phys_out,
                           wall_clamp);
  }
}

}  // namespace snap
