#pragma once

// base
#include <configure.h>

#define INP(n) (inp[(n) * stride])
#define SQR(x) ((x) * (x))

namespace snap {

// polynomial
template <typename T, int N>
DISPATCH_MACRO void interp_poly_impl(T *out, T *inp, T *coeff, int stride) {
  (*out) = 0.;
  for (int i = 0; i < N; ++i) {
    (*out) += coeff[i] * INP(i);
  }
};

// WENO 3 interpolation
template <typename T>
DISPATCH_MACRO void interp_weno3_impl(T *out, T *inp, T *c1, T *c2, T *c3,
                                      T *c4, int stride, double scale) {
  T p0 = c1[1] * phi + c1[0] * phim1;
  T p1 = c2[2] * phip1 + c2[1] * phi;

  T beta0 = SQR(c3[0] * phim1 + c3[1] * phi);
  T beta1 = SQR(c4[1] * phi + c4[2] * phip1);

  T alpha0 = (1.0 / 3.0) / SQR(beta0 + 1e-6);
  T alpha1 = (2.0 / 3.0) / SQR(beta1 + 1e-6);

  (*out) = (alpha0 * p0 + alpha1 * p1) / (alpha0 + alpha1);
};

// WENO 5 interpolation
template <typename T>
DISPATCH_MACRO T interp_weno5(T *out, T *inp, T *c1, T *c2, T *c3, T *c4, T *c5,
                              T *c6, T *c7, T *c8, T *c9, int stride,
                              double scale) {
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

  (*out) =
      (alpha0 * p0 + alpha1 * p1 + alpha2 * p2) / (alpha0 + alpha1 + alpha2);
};

}  // namespace snap

#undef SQR
#undef INP
