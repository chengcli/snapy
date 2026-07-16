#pragma once

// base
#include <configure.h>

// snap
#include <snap/snap.h>

#include "../eos/ideal_moist_impl.h"

#define SQR(x) ((x) * (x))
#define WL(n) (wl[(n) * stride_w])
#define WR(n) (wr[(n) * stride_w])
#define FLX(n) (flx[(n) * stride_f])

namespace snap {

template <typename T>
inline DISPATCH_MACRO T roe_max(T a, T b) {
  return a > b ? a : b;
}

template <typename T>
void DISPATCH_MACRO roe_impl(T* flx, T const* wl, T const* wr, T el, T er,
                             T gammal, T gammar, T cl, T cr, int dim, int ny,
                             bool ideal_moist, int nvapor, T gammad,
                             T const* inv_mu_ratio_m1, T const* cv_ratio_m1,
                             T const* u0, int stride_w, int stride_f,
                             T* face_pressure = nullptr) {
  auto TINY_NUMBER = T(1.0e-10);

  auto ivx = IPR - dim;
  auto ivy = IVX + ((ivx - IVX) + 1) % 3;
  auto ivz = IVX + ((ivx - IVX) + 2) % 3;

  T etl = el + T(0.5) * WL(IDN) * (SQR(WL(IVX)) + SQR(WL(IVY)) + SQR(WL(IVZ)));
  T etr = er + T(0.5) * WR(IDN) * (SQR(WR(IVX)) + SQR(WR(IVY)) + SQR(WR(IVZ)));

  T sqrtdl = sqrt(WL(IDN));
  T sqrtdr = sqrt(WR(IDN));
  T isdlpdr = T(1) / (sqrtdl + sqrtdr);

  T rhobar = sqrtdl * sqrtdr;
  T v1 = (sqrtdl * WL(ivx) + sqrtdr * WR(ivx)) * isdlpdr;
  T v2 = (sqrtdl * WL(ivy) + sqrtdr * WR(ivy)) * isdlpdr;
  T v3 = (sqrtdl * WL(ivz) + sqrtdr * WR(ivz)) * isdlpdr;
  T hl = (etl + WL(IPR)) / WL(IDN);
  T hr = (etr + WR(IPR)) / WR(IDN);
  T h = (hl * sqrtdl + hr * sqrtdr) * isdlpdr;

  T dryl = T(1);
  T dryr = T(1);
  for (int n = 0; n < ny; ++n) {
    dryl -= WL(ICY + n);
    dryr -= WR(ICY + n);
  }

  T rhol0 = WL(IDN) * dryl;
  T rhor0 = WR(IDN) * dryr;

  T fl0 = rhol0 * WL(ivx);
  T fr0 = rhor0 * WR(ivx);

  T fl1 = WL(IDN) * WL(ivx) * WL(ivx) + WL(IPR);
  T fr1 = WR(IDN) * WR(ivx) * WR(ivx) + WR(IPR);

  T fl2 = WL(IDN) * WL(ivx) * WL(ivy);
  T fr2 = WR(IDN) * WR(ivx) * WR(ivy);

  T fl3 = WL(IDN) * WL(ivx) * WL(ivz);
  T fr3 = WR(IDN) * WR(ivx) * WR(ivz);

  T fl4 = (etl + WL(IPR)) * WL(ivx);
  T fr4 = (etr + WR(IPR)) * WR(ivx);

  T qbar[64];
  qbar[0] = (rhol0 / sqrtdl + rhor0 / sqrtdr) * isdlpdr;
  for (int n = 0; n < ny; ++n) {
    auto rholn = WL(IDN) * WL(ICY + n);
    auto rhorn = WR(IDN) * WR(ICY + n);
    qbar[1 + n] = (rholn / sqrtdl + rhorn / sqrtdr) * isdlpdr;
  }

  T gamma_roe = T(0.5) * (gammal + gammar);
  T offset = T(0);
  if (ideal_moist) {
    T feps = ideal_moist_feps(qbar + 1, ny, nvapor, inv_mu_ratio_m1);
    T fsig = ideal_moist_fsig(qbar + 1, ny, cv_ratio_m1);
    gamma_roe = T(1) + (gammad - T(1)) * feps / fsig;

    offset = qbar[0] * u0[0];
    for (int n = 0; n < ny; ++n) {
      offset += qbar[1 + n] * u0[1 + n];
    }
  }

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
  for (int n = 0; n < ny; ++n) {
    auto rholn = WL(IDN) * WL(ICY + n);
    auto rhorn = WR(IDN) * WR(ICY + n);
    FLX(ICY + n) = T(0.5) * (rholn * WL(ivx) + rhorn * WR(ivx));
  }

  T vsq = v1 * v1 + v2 * v2 + v3 * v3;
  T gm1_roe = gamma_roe - T(1);
  T q = h - T(0.5) * vsq - offset;
  T cs_sq = roe_max(gm1_roe * roe_max(q, TINY_NUMBER), TINY_NUMBER);
  T cs = sqrt(cs_sq);
  if (face_pressure != nullptr) {
    *face_pressure =
        T(0.5) * (WL(IPR) + WR(IPR) + rhobar * cs * (WL(ivx) - WR(ivx)));
  }

  T lam_m = v1 - cs;
  T lam_0 = v1;
  T lam_p = v1 + cs;
  T spd_m = abs(lam_m);
  T spd_0 = abs(lam_0);
  T spd_p = abs(lam_p);

  bool llf_flag = false;
  T du = WR(ivx) - WL(ivx);
  T dv = WR(ivy) - WL(ivy);
  T dw = WR(ivz) - WL(ivz);
  T dp = WR(IPR) - WL(IPR);

  T alpha_m = -T(0.5) * rhobar / cs * du + T(0.5) * dp / cs_sq;
  T alpha_p = +T(0.5) * rhobar / cs * du + T(0.5) * dp / cs_sq;
  T alpha_v = rhobar * dv;
  T alpha_w = rhobar * dw;

  T alpha0 = T(0);
  T alpha_comp0 = rhor0 - rhol0 - dp / cs_sq * qbar[0];
  T rho_first = rhol0 + alpha_m * qbar[0];
  if (rho_first < T(0)) llf_flag = true;
  T rho_second = rho_first + alpha_comp0;
  if (rho_second < T(0)) llf_flag = true;
  alpha0 += alpha_comp0;
  FLX(IDN) -= T(0.5) * (spd_m * alpha_m * qbar[0] + spd_0 * alpha_comp0 +
                        spd_p * alpha_p * qbar[0]);

  for (int n = 0; n < ny; ++n) {
    auto rholn = WL(IDN) * WL(ICY + n);
    auto rhorn = WR(IDN) * WR(ICY + n);
    auto alpha_n = rhorn - rholn - dp / cs_sq * qbar[1 + n];
    auto rho1 = rholn + alpha_m * qbar[1 + n];
    if (rho1 < T(0)) llf_flag = true;
    auto rho2 = rho1 + alpha_n;
    if (rho2 < T(0)) llf_flag = true;
    alpha0 += alpha_n;
    FLX(ICY + n) -= T(0.5) * (spd_m * alpha_m * qbar[1 + n] + spd_0 * alpha_n +
                              spd_p * alpha_p * qbar[1 + n]);
  }

  FLX(ivx) -= T(0.5) * (spd_m * alpha_m * (v1 - cs) + spd_0 * (v1 * alpha0) +
                        spd_p * alpha_p * (v1 + cs));
  FLX(ivy) -= T(0.5) * (spd_m * alpha_m * v2 + spd_0 * (v2 * alpha0 + alpha_v) +
                        spd_p * alpha_p * v2);
  FLX(ivz) -= T(0.5) * (spd_m * alpha_m * v3 + spd_0 * (v3 * alpha0 + alpha_w) +
                        spd_p * alpha_p * v3);
  FLX(IPR) -=
      T(0.5) * (spd_m * alpha_m * (h - v1 * cs) +
                spd_0 * (rhobar * (hr - hl - v1 * du) + alpha0 * h - dp) +
                spd_p * alpha_p * (h + v1 * cs));

  if (lam_m > T(0)) {
    FLX(IDN) = fl0;
    FLX(ivx) = fl1;
    FLX(ivy) = fl2;
    FLX(ivz) = fl3;
    FLX(IPR) = fl4;
    for (int n = 0; n < ny; ++n) {
      FLX(ICY + n) = WL(IDN) * WL(ICY + n) * WL(ivx);
    }
  }
  if (lam_p < T(0)) {
    FLX(IDN) = fr0;
    FLX(ivx) = fr1;
    FLX(ivy) = fr2;
    FLX(ivz) = fr3;
    FLX(IPR) = fr4;
    for (int n = 0; n < ny; ++n) {
      FLX(ICY + n) = WR(IDN) * WR(ICY + n) * WR(ivx);
    }
  }

  if (llf_flag) {
    T a = T(0.5) * roe_max(abs(WL(ivx)) + cl, abs(WR(ivx)) + cr);
    FLX(IDN) = T(0.5) * (fl0 + fr0) - a * (rhor0 - rhol0);
    FLX(ivx) = T(0.5) * (fl1 + fr1) - a * du1;
    FLX(ivy) = T(0.5) * (fl2 + fr2) - a * du2;
    FLX(ivz) = T(0.5) * (fl3 + fr3) - a * du3;
    FLX(IPR) = T(0.5) * (fl4 + fr4) - a * du4;
    for (int n = 0; n < ny; ++n) {
      auto rholn = WL(IDN) * WL(ICY + n);
      auto rhorn = WR(IDN) * WR(ICY + n);
      FLX(ICY + n) =
          T(0.5) * (rholn * WL(ivx) + rhorn * WR(ivx)) - a * (rhorn - rholn);
    }
  }
}

}  // namespace snap

#undef FLX
#undef WR
#undef WL
#undef SQR
