#include <cuda_runtime.h>

#define INP(j, i) (inp[(j) * stride2 + (i) * stride1])
#define OUT(j) (out[(j) * stride_out])
#define SQR(x) ((x) * (x))

namespace snap {

template <int N, typename T>
__device__ inline T _vvdot(T *v1, T *v2) {
  T out = 0.;
  for (int i = 0; i < N; ++i) {
    out += v1[i] * v2[i];
  }
  return out;
}

// weno3
template <typename T>
__device__ void interp_weno3_impl(T *out, T *inp, T *coeff, int dim, int ndim,
                                 int nvar, int stride1, int stride2, int stride_out,
                                 bool scale, T *smem) {
  unsigned int idx[3] = {threadIdx.z, threadIdx.y, threadIdx.x};
  unsigned int len[3] = {blockDim.z, blockDim.y, blockDim.x};

  printf("idx: %u %u %u, len: %u %u %u\n", idx[0], idx[1], idx[2],
         len[0], len[1], len[2]);
  printf("block.x: %d, block.y: %d, block.z: %d\n",
         blockIdx.x, blockIdx.y, blockIdx.z);
  int idim = 3 + dim - ndim;

  // Load input into shared memory
  T *sinp = smem;
  for (int j = 0; j < nvar; ++j) {
    sinp[idx[idim] + j * len[idim]] = INP(j, idx[idim]);
  }

  // Load coefficient into shared memory
  T *scoeff = smem + len[idim] * nvar;
  constexpr int N = 12; // Number of coefficients for WENO3
  for (int i = idx[idim]; i < N; i += len[idim]) {
    scoeff[i] = coeff[i];
  }

  __syncthreads();

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
    int i = idx[idim] + j * len[idim];
    T vscale = scale ? (fabs(sinp[i]) + fabs(sinp[i+1]) + fabs(sinp[i+2])
        + fabs(sinp[i+3]) + fabs(sinp[i+4])) / 5.0 : 1.0;

    if (vscale != 0.0) {
      for (int k = 0; k < 5; ++k)
        phi[k]  = sin[i + k];
    } else {
      OUT(j) = 0.0;
      continue;
    }

    T p0 = _vvdot<5>(phi, c1);
    T p1 = _vvdot<5>(phi, c2);
    T p2 = _vvdot<5>(phi, c3);

    T beta0 = 13. / 12. * SQR(_vvdot<5>(phi, c4)) + .25 * SQR(_vvdot<5>(phi, c5));
    T beta1 = 13. / 12. * SQR(_vvdot<5>(phi, c6)) + .25 * SQR(_vvdot<5>(phi, c7));
    T beta2 = 13. / 12. * SQR(_vvdot<5>(phi, c8)) + .25 * SQR(_vvdot<5>(phi, c9));

    T alpha0 = .3 / SQR(beta0 + 1e-6);
    T alpha1 = .6 / SQR(beta1 + 1e-6);
    T alpha2 = .1 / SQR(beta2 + 1e-6);

    OUT(j) =
        (alpha0 * p0 + alpha1 * p1 + alpha2 * p2) / (alpha0 + alpha1 + alpha2);

  }
};

} // namespace snap

#undef SQR
#undef INP
#undef OUT
