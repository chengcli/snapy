#pragma once

// base
#include <configure.h>

// snap
#include <snap/snap.h>

#include "ideal_moist_impl.h"
#include "shallow_water_impl.h"

namespace snap {

template <typename T>
inline DISPATCH_MACRO void eos_side_quantities(
    T const* wl, T const* wr, int ny, FusedEos eos, T gammad,
    T const* inv_mu_ratio_m1, T const* cv_ratio_m1, T const* u0, T* el, T* er,
    T* gl, T* gr, T* cl, T* cr) {
  if (eos == FusedEos::IdealGas) {
    *el = wl[IPR] / (gammad - 1.);
    *er = wr[IPR] / (gammad - 1.);
    *gl = gammad;
    *gr = gammad;
  } else if (eos == FusedEos::ShallowWater) {
    *el = 0.;
    *er = 0.;
    *gl = 0.;
    *gr = 0.;
    shallow_water_side_quantities(wl[IDN], wr[IDN], cl, cr);
    return;
  } else {
    T yl[32], yr[32];
    T suml = 0., sumr = 0.;
    for (int n = 0; n < ny; ++n) {
      yl[n] = wl[ICY + n];
      yr[n] = wr[ICY + n];
      suml += yl[n];
      sumr += yr[n];
    }
    T fepsl = ideal_moist_feps(yl, ny, inv_mu_ratio_m1);
    T fepsr = ideal_moist_feps(yr, ny, inv_mu_ratio_m1);
    T fsigl = ideal_moist_fsig(yl, ny, cv_ratio_m1);
    T fsigr = ideal_moist_fsig(yr, ny, cv_ratio_m1);
    *el = wl[IPR] * fsigl / fepsl / (gammad - 1.);
    *er = wr[IPR] * fsigr / fepsr / (gammad - 1.);
    *el += wl[IDN] * (1. - suml) * u0[0];
    *er += wr[IDN] * (1. - sumr) * u0[0];
    for (int n = 0; n < ny; ++n) {
      *el += wl[IDN] * yl[n] * u0[1 + n];
      *er += wr[IDN] * yr[n] * u0[1 + n];
    }
    *gl = 1. + (gammad - 1.) * fepsl / fsigl;
    *gr = 1. + (gammad - 1.) * fepsr / fsigr;
  }
  *cl = sqrt((*gl) * wl[IPR] / wl[IDN]);
  *cr = sqrt((*gr) * wr[IPR] / wr[IDN]);
}

}  // namespace snap
