#pragma once

// base
#include <configure.h>

// snap
#include <snap/snap.h>

#define WL(n) (wli[(n) * stride])
#define WR(n) (wri[(n) * stride])
#define FLX(n) (flx[(n) * stride])
#define GAMMAD (*gammad)
#define CV_RATIO_M1(n) (cv_ratio_m1[(n) * stride])
#define MU_RATIO_M1(n) (mu_ratio_m1[(n) * stride])

namespace snap {

template <typename T>
void DISPATCH_MACRO lmars_impl(T *flx, T *wli, T *wri, T *gammad,
                               T *cv_ratio_m1, T *mu_ratio_m1, int dim,
                               int nvapor, int ncloud, int stride) {
  constexpr int ICY = Index::ICY;
  constexpr int IDN = Index::IDN;
  constexpr int IPR = Index::IPR;
  constexpr int IVX = Index::IVX;

  auto ivx = IPR - dim;
  auto ivy = IVX + ((ivx - IVX) + 1) % 3;
  auto ivz = IVX + ((ivx - IVX) + 2) % 3;

  // Compute kappal (for the left state)
  T fsig = 1.0;
  T feps = 1.0;
  for (int n = 0; n < nvapor; n++) {
    fsig += WL(ICY + n) * CV_RATIO_M1(n);
    feps += WL(ICY + n) * MU_RATIO_M1(n);
  }
  for (int n = nvapor; n < nvapor + ncloud; n++) {
    fsig += WL(ICY + nvapor + n) * CV_RATIO_M1(n);
    fsig -= WL(ICY + nvapor + n);
  }
  auto kappal = 1.0 / (GAMMAD - 1.0) * (fsig / feps);

  // Compute kappar (for the right state)
  fsig = 1.0;
  feps = 1.0;
  for (int n = 0; n < nvapor; n++) {
    fsig += WR(ICY + n) * CV_RATIO_M1(n);
    feps += WR(ICY + n) * MU_RATIO_M1(n);
  }
  for (int n = nvapor; n < nvapor + ncloud; n++) {
    fsig += WR(ICY + nvapor + n) * CV_RATIO_M1(n);
    fsig -= WR(ICY + nvapor + n);
  }
  auto kappar = 1.0 / (GAMMAD - 1.0) * (fsig / feps);

  // Enthalpies
  auto hl = (WL(IPR) / WL(IDN)) * (kappal + 1.0) +
            0.5 * (WL(ivx) * WL(ivx) + WL(ivy) * WL(ivy) + WL(ivz) * WL(ivz));

  auto hr = (WR(IPR) / WR(IDN)) * (kappar + 1.0) +
            0.5 * (WR(ivx) * WR(ivx) + WR(ivy) * WR(ivy) + WR(ivz) * WR(ivz));

  // Average density, average sound speed, pressure, velocity
  auto rhobar = 0.5 * (WL(IDN) + WR(IDN));
  auto cbar = sqrt(0.5 * (1.0 + (1.0 / kappar + 1.0 / kappal) / 2.0) *
                   (WL(IPR) + WR(IPR)) / rhobar);

  auto pbar =
      0.5 * (WL(IPR) + WR(IPR)) + 0.5 * (rhobar * cbar) * (WL(ivx) - WR(ivx));

  auto ubar =
      0.5 * (WL(ivx) + WR(ivx)) + 0.5 / (rhobar * cbar) * (WL(IPR) - WR(IPR));

  // Compute fluxes depending on the sign of ubar
  T rd = 1.0;
  if (ubar > 0.0) {
    // Left side flux
    for (int n = 0; n < nvapor + ncloud; n++) {
      rd -= WL(ICY + n);
    }

    FLX(IDN) = ubar * WL(IDN) * rd;
    for (int n = 0; n < nvapor + ncloud; n++) {
      FLX(ICY + n) = ubar * WL(IDN) * WL(ICY + n);
    }

    FLX(ivx) = ubar * WL(IDN) * WL(ivx) + pbar;
    FLX(ivy) = ubar * WL(IDN) * WL(ivy);
    FLX(ivz) = ubar * WL(IDN) * WL(ivz);
    FLX(IPR) = ubar * WL(IDN) * hl;
  } else {
    // Right side flux
    for (int n = 0; n < nvapor + ncloud; n++) {
      rd -= WR(ICY + n);
    }

    FLX(IDN) = ubar * WR(IDN) * rd;
    for (int n = 0; n < nvapor + ncloud; n++) {
      FLX(ICY + n) = ubar * WR(IDN) * WR(ICY + n);
    }

    FLX(ivx) = ubar * WR(IDN) * WR(ivx) + pbar;
    FLX(ivy) = ubar * WR(IDN) * WR(ivy);
    FLX(ivz) = ubar * WR(IDN) * WR(ivz);
    FLX(IPR) = ubar * WR(IDN) * hr;
  }
}

}  // namespace snap

#undef WL
#undef WR
#undef FLX
#undef GAMMAD
#undef CV_RATIO_M1
#undef MU_RATIO_M1
