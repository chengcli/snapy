#pragma once

// base
#include <configure.h>

// snap
#include <snap/snap.h>

namespace snap {

template <typename T>
inline DISPATCH_MACRO void primitive_projector_impl(
    T const* w, T* wp, T* psf, T const* dx1f, int nvar, int nc3, int nc2,
    int nc1, int col, int is, int ie, FusedPrimitiveProjector projector, T grav,
    T margin, T gas_constant) {
  int k = col / nc2;
  int j = col - k * nc2;
  int ncells = nc1 * nc2 * nc3;
  int base = k * nc2 * nc1 + j * nc1;
  int psf_base = k * nc2 * (nc1 + 1) + j * (nc1 + 1);

  for (int v = 0; v < nvar; ++v) {
    for (int i = 0; i < nc1; ++i) {
      wp[v * ncells + base + i] = w[v * ncells + base + i];
    }
  }

  for (int i = 0; i < ie; ++i) {
    psf[psf_base + i] = grav * w[IDN * ncells + base + i] * dx1f[i];
  }
  for (int i = 0; i < is; ++i) {
    psf[psf_base + i] = -psf[psf_base + i];
  }

  T rho_top = w[IDN * ncells + base + ie - 1];
  T p_top = w[IPR * ncells + base + ie - 1];
  T rd_tv = p_top / rho_top;
  psf[psf_base + ie] = p_top * exp(-grav * dx1f[ie - 1] / (T(2) * rd_tv));

  for (int i = ie + 1; i < nc1 + 1; ++i) {
    psf[psf_base + i] = grav * w[IDN * ncells + base + i - 1] * dx1f[i - 1];
  }

  for (int i = ie - 1; i >= 0; --i) {
    psf[psf_base + i] += psf[psf_base + i + 1];
  }
  for (int i = ie + 1; i < nc1 + 1; ++i) {
    psf[psf_base + i] += psf[psf_base + i - 1];
  }

  for (int i = 0; i < nc1; ++i) {
    T pl = psf[psf_base + i];
    T pr = psf[psf_base + i + 1];
    T df = pl - pr;
    T psv = fabs(df) < margin ? T(0.5) * (pl + pr) : df / log(pl / pr);
    wp[IPR * ncells + base + i] = w[IPR * ncells + base + i] - psv;
    if (projector == FusedPrimitiveProjector::Temperature) {
      wp[IDN * ncells + base + i] = w[IPR * ncells + base + i] /
                                    (w[IDN * ncells + base + i] * gas_constant);
    }
  }
}

}  // namespace snap
