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
  auto c1 = scoeff;
  auto c2 = c1 + 3;
  auto c3 = c2 + 3;
  auto c4 = c3 + 3;

  T phi[3];

  for (int j = 0; j < nvar; ++j) {
    int i = idx[idim] + j * len[idim];

    phi[0] = sinp[i];
    phi[1] = sinp[i+1];
    phi[2] = sinp[i+2];

    T vscale = scale ? (fabs(phi[0]) + fabs(phi[1]) + fabs(phi[2])) / 3.0 : 1.0;

    if (vscale != 0.0) {
      phi[0] /= vscale;
      phi[1] /= vscale;
      phi[2] /= vscale;
    }

    T p0 = _vvdot<3>(phi, c1);
    T p1 = _vvdot<3>(phi, c2);

    T beta0 = SQR(_vvdot<3>(phi, c3));
    T beta1 = SQR(_vvdot<3>(phi, c4));

    T alpha0 = (1.0 / 3.0) / SQR(beta0 + 1e-6);
    T alpha1 = (2.0 / 3.0) / SQR(beta1 + 1e-6);

    OUT(j) = (alpha0 * p0 + alpha1 * p1) / (alpha0 + alpha1) * vscale;
  }
};

} // namespace snap

#undef SQR
#undef INP
#undef OUT
