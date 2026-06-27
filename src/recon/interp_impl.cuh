#pragma once

#include <cuda_runtime.h>

#define INP(j, i) (inp[(j) * stride_in2 + (i) * stride_in1])
#define OUT(j, i) (out[(j) * stride_out2 + (i) * stride_out1])
#define SQR(x) ((x) * (x))

namespace snap {

template <int N, typename T>
inline __device__ T _vvdot(T *v1, T *v2) {
  T out = 0.;
  for (int i = 0; i < N; ++i) {
    out += v1[i] * v2[i];
  }
  return out;
}

// polynomial
template <typename T, int N>
__device__ void interp_poly_impl(T *out, T *inp, T *coeff, int nvar,
                                 int stride_in1, int stride_in2,
                                 int stride_out1, int stride_out2, T *smem) {
  int id = threadIdx.x;
  int nt = blockDim.x;

  // Load input into shared memory
  T *sinp = smem;
  for (int j = 0; j < nvar; ++j) {
    sinp[id + j * nt] = INP(j, id);
  }

  // Load coefficient into shared memory
  T *scoeff = smem + nt * nvar;
  for (int i = id; i < N; i += nt) {
    scoeff[i] = coeff[i];
  }

  bool active = id <= nt - N;

  __syncthreads();

  // drop last few threads
  if (!active) return;

  // calculation
  for (int j = 0; j < nvar; ++j) {
    int i = id + j * nt;
    T sout = 0.;

    for (int k = 0; k < N; ++k) {
      sout += scoeff[k] * sinp[i + k];
    }

    // copy to global memory
    OUT(j, id) = sout;
  }
};

// weno3
template <typename T>
__device__ void interp_weno3_impl(T *out, T *inp, T *coeff, int nvar,
                                  int stride_in1, int stride_in2,
                                  int stride_out1, int stride_out2, bool scale,
                                  T *smem) {
  int id = threadIdx.x;
  int nt = blockDim.x;

  // Load input into shared memory
  T *sinp = smem;
  for (int j = 0; j < nvar; ++j) {
    sinp[id + j * nt] = INP(j, id);
  }

  // Load coefficient into shared memory
  T *scoeff = smem + nt * nvar;
  constexpr int N = 12;  // Number of coefficients for WENO3
  for (int i = id; i < N; i += nt) {
    scoeff[i] = coeff[i];
  }

  bool active = id <= nt - 3;

  __syncthreads();

  // drop last few threads
  if (!active) return;

  // calculation
  T *c1 = scoeff;
  T *c2 = c1 + 3;
  T *c3 = c2 + 3;
  T *c4 = c3 + 3;

  T phi[3];

  for (int j = 0; j < nvar; ++j) {
    int i = id + j * nt;
    T vscale =
        scale ? (fabs(sinp[i]) + fabs(sinp[i + 1]) + fabs(sinp[i + 2])) / 3.0
              : 1.0;

    if (vscale != 0.0) {
      phi[0] = sinp[i] / vscale;
      phi[1] = sinp[i + 1] / vscale;
      phi[2] = sinp[i + 2] / vscale;
    } else {
      OUT(j, id) = 0.0;
      continue;
    }

    T p0 = _vvdot<3>(phi, c1);
    T p1 = _vvdot<3>(phi, c2);

    T beta0 = SQR(_vvdot<3>(phi, c3));
    T beta1 = SQR(_vvdot<3>(phi, c4));

    T alpha0 = (1.0 / 3.0) / SQR(beta0 + 1e-6);
    T alpha1 = (2.0 / 3.0) / SQR(beta1 + 1e-6);

    OUT(j, id) = (alpha0 * p0 + alpha1 * p1) / (alpha0 + alpha1) * vscale;
  }
};

// weno5
template <typename T>
__device__ void interp_weno5_impl(T *out, T *inp, T *coeff, int nvar,
                                  int stride_in1, int stride_in2,
                                  int stride_out1, int stride_out2, bool scale,
                                  T *smem) {
  int id = threadIdx.x;
  int nt = blockDim.x;

  // Load input into shared memory
  T *sinp = smem;
  for (int j = 0; j < nvar; ++j) {
    sinp[id + j * nt] = INP(j, id);
  }

  // Load coefficient into shared memory
  T *scoeff = smem + nt * nvar;
  constexpr int N = 45;  // Number of coefficients for WENO5
  for (int i = id; i < N; i += nt) {
    scoeff[i] = coeff[i];
  }

  bool active = id <= nt - 5;

  __syncthreads();

  // drop last few threads
  if (!active) return;

  // first thread print shared memory array
  // if (id == 0) {
  // for (int i = 0; i < nvar * nt + N; ++i)
  //   printf("smem[%d] = %f\n", i, smem[i]);
  //}

  // calculation
  T *c1 = scoeff;
  T *c2 = c1 + 5;
  T *c3 = c2 + 5;
  T *c4 = c3 + 5;
  T *c5 = c4 + 5;
  T *c6 = c5 + 5;
  T *c7 = c6 + 5;
  T *c8 = c7 + 5;
  T *c9 = c8 + 5;

  T phi[5];

  for (int j = 0; j < nvar; ++j) {
    int i = id + j * nt;
    T vscale = scale ? (fabs(sinp[i]) + fabs(sinp[i + 1]) + fabs(sinp[i + 2]) +
                        fabs(sinp[i + 3]) + fabs(sinp[i + 4])) /
                           5.0
                     : 1.0;

    if (vscale != 0.0) {
      for (int k = 0; k < 5; ++k) {
        phi[k] = sinp[i + k] / vscale;
      }
    } else {
      OUT(j, id) = 0.0;
      continue;
    }

    T p0 = _vvdot<5>(phi, c1);
    T p1 = _vvdot<5>(phi, c2);
    T p2 = _vvdot<5>(phi, c3);

    T beta0 =
        13. / 12. * SQR(_vvdot<5>(phi, c4)) + .25 * SQR(_vvdot<5>(phi, c5));
    T beta1 =
        13. / 12. * SQR(_vvdot<5>(phi, c6)) + .25 * SQR(_vvdot<5>(phi, c7));
    T beta2 =
        13. / 12. * SQR(_vvdot<5>(phi, c8)) + .25 * SQR(_vvdot<5>(phi, c9));

    T alpha0 = .3 / SQR(beta0 + 1e-6);
    T alpha1 = .6 / SQR(beta1 + 1e-6);
    T alpha2 = .1 / SQR(beta2 + 1e-6);

    OUT(j, id) = vscale * (alpha0 * p0 + alpha1 * p1 + alpha2 * p2) /
                 (alpha0 + alpha1 + alpha2);
  }
};

template <typename T>
__device__ T interp_shared_poly3_impl(T const *line, int v, int start,
                                      int axis_size, bool right) {
  constexpr T cm[3] = {-1. / 3., 5. / 6., -1. / 6.};
  T out = 0.;
  for (int k = 0; k < 3; ++k) {
    out += (right ? cm[2 - k] : cm[k]) * line[v * axis_size + start + k];
  }
  return out;
}

template <typename T>
__device__ T interp_shared_poly5_impl(T const *line, int v, int start,
                                      int axis_size, bool right) {
  constexpr T cm[5] = {-1. / 20., 9. / 20., 47. / 60., -13. / 60., 1. / 30.};
  T out = 0.;
  for (int k = 0; k < 5; ++k) {
    out += (right ? cm[4 - k] : cm[k]) * line[v * axis_size + start + k];
  }
  return out;
}

template <typename T>
__device__ T interp_shared_weno3_impl(T const *line, int v, int start,
                                      int axis_size, bool right) {
  constexpr T cm[4][3] = {{1. / 2., 1. / 2., 0.},
                          {0., 3. / 2., -1. / 2.},
                          {1., -1., 0.},
                          {0., 1., -1.}};
  T c[4][3];
  for (int r = 0; r < 4; ++r) {
    for (int k = 0; k < 3; ++k) c[r][k] = right ? cm[r][2 - k] : cm[r][k];
  }
  T phi[3];
  for (int k = 0; k < 3; ++k) {
    phi[k] = line[v * axis_size + start + k];
  }
  T p0 = _vvdot<3>(phi, c[0]);
  T p1 = _vvdot<3>(phi, c[1]);
  T beta0 = SQR(_vvdot<3>(phi, c[2]));
  T beta1 = SQR(_vvdot<3>(phi, c[3]));
  T alpha0 = (1.0 / 3.0) / SQR(beta0 + 1.e-6);
  T alpha1 = (2.0 / 3.0) / SQR(beta1 + 1.e-6);
  return (alpha0 * p0 + alpha1 * p1) / (alpha0 + alpha1);
}

template <typename T>
__device__ T interp_shared_weno5_impl(T const *line, int v, int start,
                                      int axis_size, bool right) {
  constexpr T cm[9][5] = {
      {-1. / 6., 5. / 6., 1. / 3., 0., 0.},
      {0., 1. / 3., 5. / 6., -1. / 6., 0.},
      {0., 0., 11. / 6., -7. / 6., 1. / 3.},
      {1., -2., 1., 0., 0.},
      {1., -4., 3., 0., 0.},
      {0., 1., -2., 1., 0.},
      {0., -1., 0., 1., 0.},
      {0., 0., 1., -2., 1.},
      {0., 0., 3., -4., 1.}};
  T c[9][5];
  for (int r = 0; r < 9; ++r) {
    for (int k = 0; k < 5; ++k) c[r][k] = right ? cm[r][4 - k] : cm[r][k];
  }
  T phi[5];
  for (int k = 0; k < 5; ++k) {
    phi[k] = line[v * axis_size + start + k];
  }
  T p0 = _vvdot<5>(phi, c[0]);
  T p1 = _vvdot<5>(phi, c[1]);
  T p2 = _vvdot<5>(phi, c[2]);
  T beta0 = 13. / 12. * SQR(_vvdot<5>(phi, c[3])) +
            .25 * SQR(_vvdot<5>(phi, c[4]));
  T beta1 = 13. / 12. * SQR(_vvdot<5>(phi, c[5])) +
            .25 * SQR(_vvdot<5>(phi, c[6]));
  T beta2 = 13. / 12. * SQR(_vvdot<5>(phi, c[7])) +
            .25 * SQR(_vvdot<5>(phi, c[8]));
  T alpha0 = .3 / SQR(beta0 + 1.e-6);
  T alpha1 = .6 / SQR(beta1 + 1.e-6);
  T alpha2 = .1 / SQR(beta2 + 1.e-6);
  return (alpha0 * p0 + alpha1 * p1 + alpha2 * p2) /
         (alpha0 + alpha1 + alpha2);
}

}  // namespace snap

#undef SQR
#undef INP
#undef OUT
