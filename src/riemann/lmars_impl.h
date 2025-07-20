#pragma once

// base
#include <configure.h>

// snap
#include <snap/snap.h>

#define WL(n) (wl[(n) * stride])
#define WR(n) (wr[(n) * stride])
#define FLX(n) (flx[(n) * stride])
#define SQR(x) ((x) * (x))

namespace snap {

template <typename T>
void DISPATCH_MACRO lmars_impl(T *flx, T *wl, T *wr, T hl, T hr, T gammal,
                               T gammar, int dim, int ny, int stride) {
  auto ivx = IPR - dim;
  auto ivy = IVX + ((ivx - IVX) + 1) % 3;
  auto ivz = IVX + ((ivx - IVX) + 2) % 3;

  T wli[5] = {WL(IDN), WL(ivx), WL(ivy), WL(ivz), WL(IPR)};
  T wri[5] = {WR(IDN), WR(ivx), WR(ivy), WR(ivz), WR(IPR)};

  // Enthalpies
  hl += 0.5 * (SQR(wli[IVX]) + SQR(wli[IVY]) + SQR(wli[IVZ])) +
        wli[IPR] / wli[IDN];
  hr += 0.5 * (SQR(wri[IVX]) + SQR(wri[IVY]) + SQR(wri[IVZ])) +
        wri[IPR] / wri[IDN];

  // Average density, average sound speed, pressure, velocity
  auto rhobar = 0.5 * (wli[IDN] + wri[IDN]);
  auto gamma_bar = 0.5 * (gammal + gammar);
  auto cbar = sqrt(0.5 * gamma_bar * (wli[IPR] + wri[IPR]) / rhobar);

  auto pbar = 0.5 * (wli[IPR] + wri[IPR]) +
              0.5 * (rhobar * cbar) * (wli[IVX] - wri[IVX]);

  auto ubar = 0.5 * (wli[IVX] + wri[IVX]) +
              0.5 / (rhobar * cbar) * (wli[IPR] - wri[IPR]);

  // Compute fluxes depending on the sign of ubar
  T rd = 1.0;
  if (ubar > 0.0) {
    // Left side flux
    for (int n = 0; n < ny; n++) {
      rd -= WL(ICY + n);
    }

    FLX(IDN) = ubar * wli[IDN] * rd;
    for (int n = 0; n < ny; n++) {
      FLX(ICY + n) = ubar * wli[IDN] * WL(ICY + n);
    }

    FLX(ivx) = ubar * wli[IDN] * wli[IVX] + pbar;
    FLX(ivy) = ubar * wli[IDN] * wli[IVY];
    FLX(ivz) = ubar * wli[IDN] * wli[IVZ];
    FLX(IPR) = ubar * wli[IDN] * hl;
  } else {
    // Right side flux
    for (int n = 0; n < ny; n++) {
      rd -= WR(ICY + n);
    }

    FLX(IDN) = ubar * wri[IDN] * rd;
    for (int n = 0; n < ny; n++) {
      FLX(ICY + n) = ubar * wri[IDN] * WR(ICY + n);
    }

    FLX(ivx) = ubar * wri[IDN] * wri[IVX] + pbar;
    FLX(ivy) = ubar * wri[IDN] * wri[IVY];
    FLX(ivz) = ubar * wri[IDN] * wri[IVZ];
    FLX(IPR) = ubar * wri[IDN] * hr;
  }
}

}  // namespace snap

#undef WL
#undef WR
#undef FLX
#undef SQR
