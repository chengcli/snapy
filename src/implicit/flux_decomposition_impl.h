#pragma once

// C/C++
#include <cstdarg>
#include <cstdio>

// base
#include <configure.h>

// snap
#include <snap/snap.h>

#define PRIM(i) prim[(i) * stride]

namespace snap {

constexpr int ROWS = 5;
constexpr int COLS = 5;

template <typename T>
DISPATCH_MACRO void init_matrix5(T *mat, ...) {
  va_list args;
  va_start(args, cols);
  for (int i = 0; i < ROWS * COLS; ++i) {
    mat[i] = va_arg(args, T);
  }
  va_end(args);
}

template <typename T>
inline DISPATCH_MACRO T _norm2(T *v, T *cos_theta) {
  T out = 0.;
  for (int i = 0; i < 3; ++i) {
    out += v[i] * v[i];
  }
  out += 2. * (v[1] * v[2] * cos_theta[0] + v[0] * v[2] * cos_theta[1] +
               v[0] * v[1] * cos_theta[2]);
  return out;
}

//! Roe average scheme
/*
 * Flux in the interface between i-th and i+1-th cells:
 * A(i+1/2) = [sqrt(rho(i))*A(i) + sqrt(rho(i+1))*A(i+1)]/(sqrt(rho(i)) +
 * sqrt(rho(i+1)))
 */
template <typename T>
DISPATCH_MACRO void roe_average(T *prim, T const *wl, T const *wr, T el,
                                T er, /*T const *cos_theta,*/
                                int ny, int stride) {
  auto sqrtdl = sqrt(WL(IDN));
  auto sqrtdr = sqrt(WR(IDN));
  auto isdlpdr = 1.0 / (sqrtdl + sqrtdr);

  PRIM(IDN) = sqrtdl * sqrtdr;
  PRIM(IVX) = (sqrtdl * WL(IVX) + sqrtdr * WR(IVX)) * isdlpdr;
  PRIM(IVY) = (sqrtdl * WL(IVY) + sqrtdr * WR(IVY)) * isdlpdr;
  PRIM(IVZ) = (sqrtdl * WL(IVZ) + sqrtdr * WR(IVZ)) * isdlpdr;

  auto vel[3] = {WL(IVX), WL(IVY), WL(IVZ)};
  auto ver[3] = {WR(IVX), WR(IVY), WR(IVZ)};

  T cos_theta[3] = {0., 0., 0.};
  el += 0.5 * WL(IDN) * _norm2(vel, cos_theta);
  er += 0.5 * WR(IDN) * _norm2(ver, cos_theta);

  // enthalpy divided by the density.
  PRIM(IPR) = ((el + WL(IPR)) / sqrtdl + (er + WR(IPR)) / sqrtdr) * isdlpdr;
}

template <typename T>
DISPATCH_MACRO void eigen_system_impl(T *left, T *right, T *val,
                                      T const *prim, /*T const *cos_theta,*/
                                      T ie, T cs, int dir, int stride) {
  auto r = PRIM(IDN);
  auto u = PRIM(IVX + dir);
  auto v = PRIM(IVX + (IVY - IVX + dir) % 3);
  auto w = PRIM(IVX + (IVZ - IVX + dir) % 3);
  auto p = PRIM(IPR);

  T cos_theta[3] = {0., 0., 0.};
  T vel[3] = {u, v, w};

  auto ke = 0.5 * _norm2(vel, cos_theta);
  auto hp = (ie + p) / r;
  auto h = hp + ke;

  init_matrix5(left,                       //
               1., 1., 1., 0., 0.,         //
               u - cs, u, u + cs, 0., 0.,  //
               v, v, v, 1., 0.,            //
               w, w, w, 0., 1.,            //
               h - u * cs, ke, h + u * cs, v, w);

  init_matrix5(
      right,  //
      (cs * ke + u * hp) / (2. * cs * hp), (-hp - cs * u) / (2. * cs * hp),
      -v / (2. * hp), -w / (2. * hp), 1. / (2. * hp),    //
      (hp - ke) / hp, u / hp, v / hp, w / hp, -1. / hp,  //
      (cs * ke - u * hp) / (2. * cs * hp), (hp - cs * u) / (2. * cs * hp),
      -v / (2. * hp), -w / (2. * hp), 1. / (2. * hp),  //
      -v, 0., 1., 0., 0.,                              //
      -w, 0., 0., 1., 0.);

  init_matrix5(val,                     //
               u - cs, 0., 0., 0., 0.,  //
               0., u, 0., 0., 0.,       //
               0., 0., u + cs, 0., 0.,  //
               0., 0., 0., u, 0.,       //
               0., 0., 0., 0., u);
}

template <typename T>
DISPATCH_MACRO void flux_jacobian_impl(T *dfdq,
                                       T const *prim, /*T const* cos_theta,*/
                                       T gamma, int dir, int stride) {
  auto v1 = PRIM(IVX + dir);
  auto v2 = PRIM(IVX + (IVY - IVX + dir) % 3]);
  auto v3 = PRIM(IVX + (IVZ - IVX + dir) % 3]);
  auto rho = PRIM(IDN);
  auto pres = PRIM(IPR);

  T cos_theta[3] = {0., 0., 0.};
  T vel[3] = {v1, v2, v3};
  auto s2 = _norm2(vel, cos_theta);

  auto gm1 = gamma - 1;

  auto c1 = ((gm1 - 1) * s2 / 2 - (gm1 + 1) / gm1 * pres / rho) * v1;
  auto c2 = (gm1 + 1) / gm1 * pres / rho + s2 / 2 - gm1 * v1 * v1;

  init_matrix5(dfdq, 0, 1., 0., 0., 0.,  //
               gm1 * s2 / 2 - v1 * v1, (2. - gm1) * v1, -gm1 * v2, -gm1 * v3,
               gm1,                       //
               -v1 * v2, v2, v1, 0., 0.,  //
               -v1 * v3, v3, 0., v1, 0.,  //
               c1, c2, -gm1 * v2 * v1, -gm1 * v3 * v1, (gm1 + 1) * v1);
}

}  // namespace snap

#undef PRIM
