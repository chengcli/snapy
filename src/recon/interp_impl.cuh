#include <cuda_runtime.h>

#define INP(n) (inp[(n) * vstride])
#define OUT(n) (out[(n) * vstride])
#define SQR(x) ((x) * (x))

namespace snap {

// polynomial
template <typename T, int N>
__device__ void interp_poly_impl(T *out, T *inp, T *coeff, int dim,
                                 int vstride, int nvar, T *smem) {
  unsigned int idx[3] = {threadIdx.z, threadIdx.y, threadIdx.x};
  unsigned int len[3] = {blockDim.z, blockDim.y, blockDim.x};

  // Load input into shared memory
  T *sinp = smem;
#pragma unroll
  for (int j = 0; j < nvar; ++j) {
    sinp[idx[dim-1] + j * len[dim-1]] = INP(j);
  }

  // Load coefficient into shared memory
  T *scoeff = smem + len[dim-1] * nvar;
#pragma unroll
  for (int i = 0; i < N; ++i) scoeff[i] = coeff[i];

  __syncthreads();

  // calculation
  for (int j = 0; j < nvar; ++j) {
    T sout = 0.;

#pragma unroll
    for (int i = 0; i < N; ++i) {
      sout += scoeff[i] * sinp[i + j * len[dim-1]];
    }

    // copy to global memory
    OUT(j) = sout;
  }
};

} // namespace snap

#undef SQR
#undef INP
#undef OUT
