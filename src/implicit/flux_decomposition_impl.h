#pragma once

// C/C++
#include <cstdarg>
#include <cstdio>

// base
#include <configure.h>

// snap
#include <snap/snap.h>

#define SQR(x) ((x) * (x))
#define PRIM(i) prim[(i) * stride]
#define WL(i) wl[(i) * stride]
#define WR(i) wr[(i) * stride]

namespace snap {

constexpr int ROWS = 5;
constexpr int COLS = 5;

template <typename T, int N>
DISPATCH_MACRO inline void init_matrix(T *mat, const double (&values)[N]) {
#pragma unroll
  for (int i = 0; i < N; ++i) {
    mat[i] = values[i];
  }
}

//! Roe average scheme
/*
 * Flux in the interface between i-th and i+1-th cells:
 * A(i+1/2) = [sqrt(rho(i))*A(i) + sqrt(rho(i+1))*A(i+1)]/(sqrt(rho(i)) +
 * sqrt(rho(i+1)))
 */
template <typename T>
DISPATCH_MACRO void roe_average_impl(T *prim, T const *wl, T const *wr, T gamma,
                                     int stride) {
  auto sqrtdl = sqrt(WL(IDN));
  auto sqrtdr = sqrt(WR(IDN));
  auto isdlpdr = 1.0 / (sqrtdl + sqrtdr);

  PRIM(IDN) = sqrtdl * sqrtdr;
  PRIM(IVX) = (sqrtdl * WL(IVX) + sqrtdr * WR(IVX)) * isdlpdr;
  PRIM(IVY) = (sqrtdl * WL(IVY) + sqrtdr * WR(IVY)) * isdlpdr;
  PRIM(IVZ) = (sqrtdl * WL(IVZ) + sqrtdr * WR(IVZ)) * isdlpdr;

  auto gm1 = gamma - 1;

  auto el = WL(IPR) / gm1 +
            0.5 * WL(IDN) * (SQR(WL(IVX)) + SQR(WL(IVY)) + SQR(WL(IVZ)));

  auto er = WR(IPR) / gm1 +
            0.5 * WR(IDN) * (SQR(WR(IVX)) + SQR(WR(IVY)) + SQR(WR(IVZ)));

  // enthalpy divided by the density.
  auto hbar = ((el + WL(IPR)) / sqrtdl + (er + WR(IPR)) / sqrtdr) * isdlpdr;

  // Roe averaged pressure
  PRIM(IPR) =
      (hbar - 0.5 * (SQR(PRIM(IVX)) + SQR(PRIM(IVY)) + SQR(PRIM(IVZ)))) * gm1 /
      (gm1 + 1.) * PRIM(IDN);
}

template <typename T>
DISPATCH_MACRO void eigen_system_impl(T *left, T *right, T *val, T const *prim,
                                      T gamma, int dim, int stride) {
  auto ivx = IPR - dim;
  auto ivy = IVX + ((ivx - IVX) + 1) % 3;
  auto ivz = IVX + ((ivx - IVX) + 2) % 3;

  auto r = PRIM(IDN);
  auto u = PRIM(ivx);
  auto v = PRIM(ivy);
  auto w = PRIM(ivz);
  auto p = PRIM(IPR);

  auto gm1 = gamma - 1.;
  auto ke = 0.5 * (SQR(u) + SQR(v) + SQR(w));
  auto hp = (gm1 + 1.) / gm1 * p / r;
  auto h = hp + ke;

  auto cs = sqrt(gamma * p / r);  // sound speed

  double arr1[] = {1.,         1., 1.,         0., 0.,  //
                   u - cs,     u,  u + cs,     0., 0.,  //
                   v,          v,  v,          1., 0.,  //
                   w,          w,  w,          0., 1.,  //
                   h - u * cs, ke, h + u * cs, v,  w};

  init_matrix(left, arr1);

  double arr2[] = {(cs * ke + u * hp) / (2. * cs * hp),
                   (-hp - cs * u) / (2. * cs * hp),
                   -v / (2. * hp),
                   -w / (2. * hp),
                   1. / (2. * hp),  //
                   (hp - ke) / hp,
                   u / hp,
                   v / hp,
                   w / hp,
                   -1. / hp,  //
                   (cs * ke - u * hp) / (2. * cs * hp),
                   (hp - cs * u) / (2. * cs * hp),
                   -v / (2. * hp),
                   -w / (2. * hp),
                   1. / (2. * hp),  //
                   -v,
                   0.,
                   1.,
                   0.,
                   0.,  //
                   -w,
                   0.,
                   0.,
                   1.,
                   0.};

  init_matrix(right, arr2);

  double arr3[] = {u - cs, 0., 0.,     0., 0.,  //
                   0.,     u,  0.,     0., 0.,  //
                   0.,     0., u + cs, 0., 0.,  //
                   0.,     0., 0.,     u,  0.,  //
                   0.,     0., 0.,     0., u};

  init_matrix(val, arr3);
}

template <typename T>
DISPATCH_MACRO void flux_jacobian_impl(T *dfdq, T const *prim, T gamma, int dim,
                                       int stride) {
  auto ivx = IPR - dim;
  auto ivy = IVX + ((ivx - IVX) + 1) % 3;
  auto ivz = IVX + ((ivx - IVX) + 2) % 3;

  auto rho = PRIM(IDN);
  auto v1 = PRIM(ivx);
  auto v2 = PRIM(ivy);
  auto v3 = PRIM(ivz);
  auto pres = PRIM(IPR);

  auto s2 = SQR(v1) + SQR(v2) + SQR(v3);
  auto gm1 = gamma - 1;

  auto c1 = ((gm1 - 1.) * s2 / 2. - (gm1 + 1.) / gm1 * pres / rho) * v1;
  auto c2 = (gm1 + 1.) / gm1 * pres / rho + s2 / 2. - gm1 * v1 * v1;

  double arr[] = {0,
                  1.,
                  0.,
                  0.,
                  0.,  //
                  gm1 * s2 / 2. - v1 * v1,
                  (2. - gm1) * v1,
                  -gm1 * v2,
                  -gm1 * v3,
                  gm1,  //
                  -v1 * v2,
                  v2,
                  v1,
                  0.,
                  0.,  //
                  -v1 * v3,
                  v3,
                  0.,
                  v1,
                  0.,  //
                  c1,
                  c2,
                  -gm1 * v2 * v1,
                  -gm1 * v3 * v1,
                  (gm1 + 1.) * v1};

  init_matrix(dfdq, arr);
}

}  // namespace snap

#undef PRIM
#undef SQR
#undef WL
#undef WR
