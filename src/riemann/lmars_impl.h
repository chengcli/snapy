#pragma once

// base
#include <configure.h>

// snap
#include <snap/snap.h>

#define WL(n) (wl[(n) * stride])
#define WR(n) (wr[(n) * stride])
#define FLX(n) (flx[(n) * stride])
#define SQR(x) ((x) * (x))
#define HL (*hl)
#define HR (*hr)
#define GAMMAL (*gammal)
#define GAMMAR (*gammar)

namespace snap {

template <typename T>
void DISPATCH_MACRO lmars_impl(T *flx, T *wl, T *wr, T *hl, T *hr, T *gammal,
                               T *gammar, int dim, int ny, int stride) {
  constexpr int ICY = Index::ICY;
  constexpr int IDN = Index::IDN;
  constexpr int IPR = Index::IPR;
  constexpr int IVX = Index::IVX;

  auto ivx = IPR - dim;
  auto ivy = IVX + ((ivx - IVX) + 1) % 3;
  auto ivz = IVX + ((ivx - IVX) + 2) % 3;

  // Enthalpies
  HL += 0.5 * (SQR(WL(ivx)) + SQR(WL(ivy)) + SQR(WL(ivz))) + WL(IPR) / WL(IDN);
  HR += 0.5 * (SQR(WR(ivx)) + SQR(WR(ivy)) + SQR(WR(ivz))) + WR(IPR) / WR(IDN);

  // Average density, average sound speed, pressure, velocity
  auto rhobar = 0.5 * (WL(IDN) + WR(IDN));
  auto gamma_bar = 0.5 * (GAMMAL + GAMMAR);
  auto cbar = sqrt(0.5 * gamma_bar * (WL(IPR) + WR(IPR)) / rhobar);

  auto pbar =
      0.5 * (WL(IPR) + WR(IPR)) + 0.5 * (rhobar * cbar) * (WL(ivx) - WR(ivx));

  auto ubar =
      0.5 * (WL(ivx) + WR(ivx)) + 0.5 / (rhobar * cbar) * (WL(IPR) - WR(IPR));

  // Compute fluxes depending on the sign of ubar
  T rd = 1.0;
  if (ubar > 0.0) {
    // Left side flux
    for (int n = 0; n < ny; n++) {
      rd -= WL(ICY + n);
    }

    FLX(IDN) = ubar * WL(IDN) * rd;
    for (int n = 0; n < ny; n++) {
      FLX(ICY + n) = ubar * WL(IDN) * WL(ICY + n);
    }

    FLX(ivx) = ubar * WL(IDN) * WL(ivx) + pbar;
    FLX(ivy) = ubar * WL(IDN) * WL(ivy);
    FLX(ivz) = ubar * WL(IDN) * WL(ivz);
    FLX(IPR) = ubar * WL(IDN) * HL;
  } else {
    // Right side flux
    for (int n = 0; n < ny; n++) {
      rd -= WR(ICY + n);
    }

    FLX(IDN) = ubar * WR(IDN) * rd;
    for (int n = 0; n < ny; n++) {
      FLX(ICY + n) = ubar * WR(IDN) * WR(ICY + n);
    }

    FLX(ivx) = ubar * WR(IDN) * WR(ivx) + pbar;
    FLX(ivy) = ubar * WR(IDN) * WR(ivy);
    FLX(ivz) = ubar * WR(IDN) * WR(ivz);
    FLX(IPR) = ubar * WR(IDN) * HR;
  }
}

}  // namespace snap

#undef WL
#undef WR
#undef FLX
#undef SQR
#undef HL
#undef HR
#undef GAMMAL
#undef GAMMAR
