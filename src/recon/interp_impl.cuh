#include <cuda_runtime.h>

#define INP(j, i) (inp[(j) * stride2 + (i) * stride1])
#define OUT(j) (out[(j) * stride2])
#define SQR(x) ((x) * (x))

namespace snap {

// polynomial
template <typename T, int N>
__device__ void interp_poly_impl(T *out, T *inp, T *coeff, int dim, int ndim,
                                 int nvar, int stride1, int stride2, T *smem) {
  unsigned int idx[3] = {threadIdx.z, threadIdx.y, threadIdx.x};
  unsigned int len[3] = {blockDim.z, blockDim.y, blockDim.x};

  int idim = 3 + dim - ndim;

  // Load input into shared memory
  T *sinp = smem;
  for (int j = 0; j < nvar; ++j) {
    sinp[idx[idim] + j * len[idim]] = INP(j, idx[idim]);
  }

  // Load coefficient into shared memory
  T *scoeff = smem + len[idim] * nvar;

  // first thread loads coefficients
  if (idx[idim] == 0) {
  #pragma unroll
    for (int i = 0; i < N; ++i) scoeff[i] = coeff[i];
  }

  __syncthreads();

  // calculation
  int count = 0;
  for (int j = 0; j < nvar; ++j) {
    T sout = 0.;

#pragma unroll
    for (int i = 0; i < N; ++i) {
      sout += scoeff[i] * sinp[count++];
    }

    // copy to global memory
    OUT(j) = sout;
  }
};

} // namespace snap

#undef SQR
#undef INP
#undef OUT
