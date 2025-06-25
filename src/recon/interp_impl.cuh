#include <cuda_runtime.h>

#define INP(n) (inp[(n) * stride])
#define SQR(x) ((x) * (x))

namespace snap {

// polynomial
template <typename T, int N>
__device__ void interp_poly_impl(T *out, T *inp, T *coeff, int dim, int stride, int nvar) {
  // Shared memory allocation
  extern __shared__ T sdata[];

  unsigned int idx[3] = {threadIdx.z, threadIdx.y, threadIdx.x};
  unsigned int len[3] = {blockDim.z, blockDim.y, blockDim.x};

  T *sinp = sdata;

  // Load coefficients into shared memory
  T *scoeff = sdata + len[dim-1];

#pragma unroll
  for (int i = 0; i < N; ++i) scoeff[i] = coeff[i];

  for (int j = 0; j < nvar; ++i) {
    // Load input into shared memory
    sinp[idx[dim-1]] = INP(j);

    __syncthreads();

    T sout = 0.;

#pragma unroll
    for (int i = 0; i < N; ++i) {
      sout += scoeff[i] * sinp[i];
    }

    // copy to global memory
    OUT(j) = sout;
  }
};

} // namespace snap

#undef SQR
#undef INP
