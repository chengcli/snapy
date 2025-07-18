#pragma once

// base
#include <configure.h>

// snap
#include <snap/snap.h>

#define SQR(x) ((x) * (x))
#define FLX(n) flx[(n) * stride]
#define WL(n) wl[(n) * stride]
#define WR(n) wr[(n) * stride]
#define EL (*el)
#define ER (*er)
#define GAMMAL (*gammal)
#define GAMMAR (*gammar)
#define CL (*cl)
#define CR (*cr)

namespace snap {

template <typename T>
void DISPATCH_MACRO hllc_impl(T *flx, T *wl, T *wr, T *el, T *er, T *gammal,
                              T *gammar, T *cl, T *cr, int dim, int ny,
                              int stride) {
  constexpr int ICY = Index::ICY;
  constexpr int IDN = Index::IDN;
  constexpr int IPR = Index::IPR;
  constexpr int IVX = Index::IVX;
  constexpr int IVY = Index::IVY;
  constexpr int IVZ = Index::IVZ;

  auto TINY_NUMBER = 1.0e-10;

  auto ivx = IPR - dim;
  auto ivy = IVX + ((ivx - IVX) + 1) % 3;
  auto ivz = IVX + ((ivx - IVX) + 2) % 3;

  //--- Step 1.  Compute kappa

  //--- Step 2.  Compute middle state estimates with PVRS (Toro 10.5.2)

  EL += 0.5 * WL(IDN) * (SQR(WL(IVX)) + SQR(WL(IVY)) + SQR(WL(IVZ)));
  ER += 0.5 * WR(IDN) * (SQR(WR(IVX)) + SQR(WR(IVY)) + SQR(WR(IVZ)));

  auto rhoa = .5 * (WL(IDN) + WR(IDN));  // average density
  auto ca = .5 * (CL + CR);              // average sound speed
  auto pmid = .5 * (WL(IPR) + WR(IPR) + (WL(ivx) - WR(ivx)) * rhoa * ca);
  auto umid = .5 * (WL(ivx) + WR(ivx) + (WL(IPR) - WR(IPR)) / (rhoa * ca));

  //--- Step 3.  Compute sound speed in L,R

  auto ql = (pmid <= WL(IPR)) ? 1.0
                              : std::sqrt(1.0 + (GAMMAL + 1) / (2 * GAMMAL) *
                                                    (pmid / WL(IPR) - 1.0));
  auto qr = (pmid <= WR(IPR)) ? 1.0
                              : std::sqrt(1.0 + (GAMMAR + 1) / (2 * GAMMAR) *
                                                    (pmid / WR(IPR) - 1.0));

  //--- Step 4.  Compute the max/min wave speeds based on L/R

  auto al = WL(ivx) - CL * ql;
  auto ar = WR(ivx) + CR * qr;

  auto bp = ar > 0.0 ? ar : (TINY_NUMBER);
  auto bm = al < 0.0 ? al : -(TINY_NUMBER);

  //--- Step 5. Compute the contact wave speed and pressure

  auto vxl = WL(ivx) - al;
  auto vxr = WR(ivx) - ar;

  auto tl = WL(IPR) + vxl * WL(IDN) * WL(ivx);
  auto tr = WR(IPR) + vxr * WR(IDN) * WR(ivx);

  auto ml = WL(IDN) * vxl;
  auto mr = -(WR(IDN) * vxr);

  // Determine the contact wave speed...
  auto am = (tl - tr) / (ml + mr);
  // ...and the pressure at the contact surface
  auto cp = (ml * tr + mr * tl) / (ml + mr);
  cp = cp > 0.0 ? cp : 0.0;

  //--- Step 6. Compute L/R fluxes along the line bm, bp

  vxl = WL(ivx) - bm;
  vxr = WR(ivx) - bp;

  T fl[ICY], fr[ICY];

  fl[IDN] = WL(IDN) * vxl;
  fr[IDN] = WR(IDN) * vxr;

  fl[ivx] = WL(IDN) * WL(ivx) * vxl + WL(IPR);
  fr[ivx] = WR(IDN) * WR(ivx) * vxr + WR(IPR);

  fl[ivy] = WL(IDN) * WL(ivy) * vxl;
  fr[ivy] = WR(IDN) * WR(ivy) * vxr;

  fl[ivz] = WL(IDN) * WL(ivz) * vxl;
  fr[ivz] = WR(IDN) * WR(ivz) * vxr;

  fl[IPR] = EL * vxl + WL(IPR) * WL(ivx);
  fr[IPR] = ER * vxr + WR(IPR) * WR(ivx);

  //--- Step 8. Compute flux weights or scales

  T sl, sr, sm;
  if (am >= 0.0) {
    sl = am / (am - bm);
    sr = 0.0;
    sm = -bm / (am - bm);
  } else {
    sl = 0.0;
    sr = -am / (bp - am);
    sm = bp / (bp - am);
  }

  //--- Step 9. Compute the HLLC flux at interface, including weighted
  // contribution
  // of the flux along the contact

  FLX(IDN) = sl * fl[IDN] + sr * fr[IDN];
  FLX(ivx) = sl * fl[ivx] + sr * fr[ivx] + sm * cp;
  FLX(ivy) = sl * fl[ivy] + sr * fr[ivy];
  FLX(ivz) = sl * fl[ivz] + sr * fr[ivz];
  FLX(IPR) = sl * fl[IPR] + sr * fr[IPR] + sm * cp * am;
}

}  // namespace snap

#undef WL
#undef WR
#undef FLX
#undef SQR
#undef EL
#undef ER
#undef GAMMAL
#undef GAMMAR
#undef CL
#undef CR
