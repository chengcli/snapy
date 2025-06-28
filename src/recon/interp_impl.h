#pragma once

// base
#include <configure.h>

#define INP(j, i) (inp[(j) * stride2 + (i) * stride1])
#define OUT(j) (out[(j) * stride_out])
#define SQR(x) ((x) * (x))

namespace snap {

template <int N, typename T>
DISPATCH_MACRO T vvdot(T *v1, T *v2) {
  T out = 0.;
  for (int i = 0; i < N; ++i) {
    out += v1[i] * v2[i];
  }
  return out;
}

// polynomial
template <typename T, int N>
DISPATCH_MACRO void interp_poly_impl(T *out, T *inp, T *coeff, int stride1,
                                     int stride2, int stride_out, int nvar) {
  for (int j = 0; j < nvar; ++j) {
    OUT(j) = 0.;
    for (int i = 0; i < N; ++i) {
      OUT(j) += coeff[i] * INP(j, i);
    }
  }
};

// WENO 3 interpolation
template <typename T>
DISPATCH_MACRO void interp_weno3_impl(T *out, T *inp, T *coeff, int stride1,
                                      int stride2, int stride_out, int nvar,
                                      double scale) {
  T *c1 = coeff;
  T *c2 = c1 + 3;
  T *c3 = c2 + 3;
  T *c4 = c3 + 3;

  T phi[3];

  for (int j = 0; j < nvar; ++j) {
    phi[0] = INP(j, 0);
    phi[1] = INP(j, 1);
    phi[2] = INP(j, 2);

    T vscale = scale ? (fabs(phi[0]) + fabs(phi[1]) + fabs(phi[2])) / 3.0 : 1.0;

    if (vscale != 0.0) {
      phi[0] /= vscale;
      phi[1] /= vscale;
      phi[2] /= vscale;
    }

    T p0 = vvdot<3>(phi, c1);
    T p1 = vvdot<3>(phi, c2);

    T beta0 = SQR(vvdot<3>(phi, c3));
    T beta1 = SQR(vvdot<3>(phi, c4));

    T alpha0 = (1.0 / 3.0) / SQR(beta0 + 1e-6);
    T alpha1 = (2.0 / 3.0) / SQR(beta1 + 1e-6);

    OUT(j) = (alpha0 * p0 + alpha1 * p1) / (alpha0 + alpha1) * vscale;
  }
};

// WENO 5 interpolation
template <typename T>
DISPATCH_MACRO void interp_weno5_impl(T *out, T *inp, T *coeff, int stride1,
                                      int stride2, int stride_out, int nvar,
                                      double scale) {
  T *c1 = coeff;
  T *c2 = c1 + 5;
  T *c3 = c2 + 5;
  T *c4 = c3 + 5;
  T *c5 = c4 + 5;
  T *c6 = c5 + 5;
  T *c7 = c6 + 5;
  T *c8 = c7 + 5;
  T *c9 = c8 + 5;
  for (int j = 0; j < nvar; ++j) {
    auto phim2 = INP(j, 0);
    auto phim1 = INP(j, 1);
    auto phi = INP(j, 2);
    auto phip1 = INP(j, 3);
    auto phip2 = INP(j, 4);

    T p0 = c1[2] * phi + c1[1] * phim1 + c1[0] * phim2;
    T p1 = c2[3] * phip1 + c2[2] * phi + c2[1] * phim1;
    T p2 = c3[4] * phip2 + c3[3] * phip1 + c3[2] * phi;

    T beta0 = 13. / 12. * SQR(c4[2] * phi + c4[1] * phim1 + c4[0] * phim2) +
              .25 * SQR(c5[2] * phi + c5[1] * phim1 + c5[0] * phim2);
    T beta1 = 13. / 12. * SQR(c6[3] * phip1 + c6[2] * phi + c6[1] * phim1) +
              .25 * SQR(c7[3] * phip1 + c7[1] * phim1);
    T beta2 = 13. / 12. * SQR(c8[4] * phip2 + c8[3] * phip1 + c8[2] * phi) +
              .25 * SQR(c9[4] * phip2 + c9[3] * phip1 + c9[2] * phi);

    T alpha0 = .3 / SQR(beta0 + 1e-6);
    T alpha1 = .6 / SQR(beta1 + 1e-6);
    T alpha2 = .1 / SQR(beta2 + 1e-6);

    OUT(j) =
        (alpha0 * p0 + alpha1 * p1 + alpha2 * p2) / (alpha0 + alpha1 + alpha2);
  }
};

}  // namespace snap

#undef OUT
#undef SQR
#undef INP
