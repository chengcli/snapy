#pragma once

// base
#include <configure.h>

// snap
#include <snap/snap.h>

#define SQR(x) ((x) * (x))
#define WL(n) (wl[(n) * stride_w])
#define WR(n) (wr[(n) * stride_w])
#define FLX(n) (flx[(n) * stride_f])

namespace snap {

template <typename T>
void DISPATCH_MACRO roe_impl(T* flx, T const* wl, T const* wr, T el, T er,
                             T gammal, T gammar, T cl, T cr, int dim,
                             int stride_w, int stride_f,
                             T* face_pressure = nullptr) {
  auto TINY_NUMBER = T(1.0e-10);

  auto ivx = IPR - dim;
  auto ivy = IVX + ((ivx - IVX) + 1) % 3;
  auto ivz = IVX + ((ivx - IVX) + 2) % 3;

  T etl = el + T(0.5) * WL(IDN) *
                   (SQR(WL(IVX)) + SQR(WL(IVY)) + SQR(WL(IVZ)));
  T etr = er + T(0.5) * WR(IDN) *
                   (SQR(WR(IVX)) + SQR(WR(IVY)) + SQR(WR(IVZ)));

  T sqrtdl = sqrt(WL(IDN));
  T sqrtdr = sqrt(WR(IDN));
  T isdlpdr = T(1) / (sqrtdl + sqrtdr);

  T d = sqrtdl * sqrtdr;
  T v1 = (sqrtdl * WL(ivx) + sqrtdr * WR(ivx)) * isdlpdr;
  T v2 = (sqrtdl * WL(ivy) + sqrtdr * WR(ivy)) * isdlpdr;
  T v3 = (sqrtdl * WL(ivz) + sqrtdr * WR(ivz)) * isdlpdr;
  T h = ((etl + WL(IPR)) / sqrtdl + (etr + WR(IPR)) / sqrtdr) * isdlpdr;
  if (face_pressure != nullptr) *face_pressure = h;

  T fl0 = WL(IDN) * WL(ivx);
  T fr0 = WR(IDN) * WR(ivx);

  T fl1 = WL(IDN) * WL(ivx) * WL(ivx) + WL(IPR);
  T fr1 = WR(IDN) * WR(ivx) * WR(ivx) + WR(IPR);

  T fl2 = WL(IDN) * WL(ivx) * WL(ivy);
  T fr2 = WR(IDN) * WR(ivx) * WR(ivy);

  T fl3 = WL(IDN) * WL(ivx) * WL(ivz);
  T fr3 = WR(IDN) * WR(ivx) * WR(ivz);

  T fl4 = (etl + WL(IPR)) * WL(ivx);
  T fr4 = (etr + WR(IPR)) * WR(ivx);

  T du0 = WR(IDN) - WL(IDN);
  T du1 = WR(IDN) * WR(ivx) - WL(IDN) * WL(ivx);
  T du2 = WR(IDN) * WR(ivy) - WL(IDN) * WL(ivy);
  T du3 = WR(IDN) * WR(ivz) - WL(IDN) * WL(ivz);
  T du4 = etr - etl;

  FLX(IDN) = T(0.5) * (fl0 + fr0);
  FLX(ivx) = T(0.5) * (fl1 + fr1);
  FLX(ivy) = T(0.5) * (fl2 + fr2);
  FLX(ivz) = T(0.5) * (fl3 + fr3);
  FLX(IPR) = T(0.5) * (fl4 + fr4);

  T vsq = v1 * v1 + v2 * v2 + v3 * v3;
  T gamma_roe = T(0.5) * (gammal + gammar);
  T gm1_roe = gamma_roe - T(1);
  T q = h - T(0.5) * vsq;
  T cs_sq = q > T(0) ? gm1_roe * q : TINY_NUMBER;
  T cs = sqrt(cs_sq);

  T ev0 = v1 - cs;
  T ev1 = v1;
  T ev2 = v1;
  T ev3 = v1;
  T ev4 = v1 + cs;

  T na = T(0.5) / cs_sq;
  T a0 = (du0 * (T(0.5) * gm1_roe * vsq + v1 * cs) -
          du1 * (gm1_roe * v1 + cs) - du2 * gm1_roe * v2 -
          du3 * gm1_roe * v3 + du4 * gm1_roe) *
         na;

  T a1 = -du0 * v2 + du2;
  T a2 = -du0 * v3 + du3;

  T qa = gm1_roe / cs_sq;
  T a3 = du0 * (T(1) - na * gm1_roe * vsq) + du1 * qa * v1 +
         du2 * qa * v2 + du3 * qa * v3 - du4 * qa;

  T a4 = (du0 * (T(0.5) * gm1_roe * vsq - v1 * cs) -
          du1 * (gm1_roe * v1 - cs) - du2 * gm1_roe * v2 -
          du3 * gm1_roe * v3 + du4 * gm1_roe) *
         na;

  T coeff0 = -T(0.5) * abs(ev0) * a0;
  T coeff1 = -T(0.5) * abs(ev1) * a1;
  T coeff2 = -T(0.5) * abs(ev2) * a2;
  T coeff3 = -T(0.5) * abs(ev3) * a3;
  T coeff4 = -T(0.5) * abs(ev4) * a4;

  bool llf_flag = false;
  T dens = WL(IDN) + a0;
  if (dens < T(0)) llf_flag = true;
  dens += a3;
  if (dens < T(0)) llf_flag = true;

  FLX(IDN) += coeff0 + coeff3 + coeff4;
  FLX(ivx) += coeff0 * (v1 - cs) + coeff3 * v1 + coeff4 * (v1 + cs);
  FLX(ivy) += coeff0 * v2 + coeff1 + coeff3 * v2 + coeff4 * v2;
  FLX(ivz) += coeff0 * v3 + coeff2 + coeff3 * v3 + coeff4 * v3;
  FLX(IPR) += coeff0 * (h - v1 * cs) + coeff1 * v2 + coeff2 * v3 +
              coeff3 * T(0.5) * vsq + coeff4 * (h + v1 * cs);

  if (ev0 > T(0)) {
    FLX(IDN) = fl0;
    FLX(ivx) = fl1;
    FLX(ivy) = fl2;
    FLX(ivz) = fl3;
    FLX(IPR) = fl4;
  }
  if (ev4 < T(0)) {
    FLX(IDN) = fr0;
    FLX(ivx) = fr1;
    FLX(ivy) = fr2;
    FLX(ivz) = fr3;
    FLX(IPR) = fr4;
  }

  if (llf_flag) {
    T a = T(0.5) * max(abs(WL(ivx)) + cl, abs(WR(ivx)) + cr);
    FLX(IDN) = T(0.5) * (fl0 + fr0) - a * du0;
    FLX(ivx) = T(0.5) * (fl1 + fr1) - a * du1;
    FLX(ivy) = T(0.5) * (fl2 + fr2) - a * du2;
    FLX(ivz) = T(0.5) * (fl3 + fr3) - a * du3;
    FLX(IPR) = T(0.5) * (fl4 + fr4) - a * du4;
  }
}

}  // namespace snap

#undef FLX
#undef WR
#undef WL
#undef SQR
