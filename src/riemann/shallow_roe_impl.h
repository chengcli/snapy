#pragma once

// snap
#include <snap/snap.h>

#include "../eos/shallow_water_impl.h"

namespace snap {

#define SWL(n) (wl[(n) * stride_w])
#define SWR(n) (wr[(n) * stride_w])
#define SFLX(n) (flx[(n) * stride_f])

template <typename T>
void DISPATCH_MACRO shallow_roe_impl(T* flx, T const* wl, T const* wr, int dim,
                                     int dir_yz, int stride_w, int stride_f) {
  int ivx, ivy, ivz;
  if (dir_yz) {
    ivx = dim == 2 ? IVY : IVZ;
    ivy = dim == 2 ? IVZ : IVY;
    ivz = IVX;
  } else {
    ivx = dim == 3 ? IVX : IVY;
    ivy = dim == 3 ? IVY : IVX;
    ivz = IVZ;
  }

  T sqrtdl = sqrt(SWL(IDN));
  T sqrtdr = sqrt(SWR(IDN));
  T isdlpdr = T(1) / (sqrtdl + sqrtdr);

  T ubar = (SWL(ivx) * sqrtdl + SWR(ivx) * sqrtdr) * isdlpdr;
  T vbar = (SWL(ivy) * sqrtdl + SWR(ivy) * sqrtdr) * isdlpdr;
  T cbar = shallow_water_roe_sound_speed(SWL(IDN), SWR(IDN));

  T del0 = SWR(IDN) - SWL(IDN);
  T delx = SWR(ivx) - SWL(ivx);
  T dely = SWR(ivy) - SWL(ivy);
  T hbar = sqrt(SWL(IDN) * SWR(IDN));

  T a1 = T(0.5) * (cbar * del0 - hbar * delx) / cbar;
  T a2 = hbar * dely;
  T a3 = T(0.5) * (cbar * del0 + hbar * delx) / cbar;

  T wave0[4] = {0, 0, 0, 0};
  T wave1[4] = {0, 0, 0, 0};
  T wave2[4] = {0, 0, 0, 0};
  wave0[IDN] = a1;
  wave0[ivx] = a1 * (ubar - cbar);
  wave0[ivy] = a1 * vbar;
  wave1[ivy] = a2;
  wave2[IDN] = a3;
  wave2[ivx] = a3 * (ubar + cbar);
  wave2[ivy] = a3 * vbar;

  T speed0 = abs(ubar - cbar);
  T speed1 = abs(ubar);
  T speed2 = abs(ubar + cbar);

  SFLX(IDN) = T(0.5) * (SWL(IDN) * SWL(ivx) + SWR(IDN) * SWR(ivx));
  SFLX(ivx) =
      T(0.5) * (SWL(IDN) * SWL(ivx) * SWL(ivx) + T(0.5) * SWL(IDN) * SWL(IDN) +
                SWR(IDN) * SWR(ivx) * SWR(ivx) + T(0.5) * SWR(IDN) * SWR(IDN));
  SFLX(ivy) = T(0.5) *
              (SWL(IDN) * SWL(ivx) * SWL(ivy) + SWR(IDN) * SWR(ivx) * SWR(ivy));
  SFLX(ivz) = 0;

  for (int v = 0; v < 4; ++v) {
    SFLX(v) -=
        T(0.5) * (speed0 * wave0[v] + speed1 * wave1[v] + speed2 * wave2[v]);
  }
}

#undef SWL
#undef SWR
#undef SFLX

}  // namespace snap
