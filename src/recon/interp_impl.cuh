#include <cuda_runtime.h>

#define INP(j, i) (inp[(j) * stride2 + (i) * stride1])
#define OUT(j) (out[(j) * stride_out])

namespace snap {

// polynomial
template <typename T, int N>
__device__ void interp_poly_impl(T *out, T *inp, T *coeff, int dim, int ndim,
                                 int nvar, int stride1, int stride2, int stride_out, T *smem) {
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
  for (int i = idx[idim]; i < N; i += len[idim]) {
    scoeff[i] = coeff[i];
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

#undef INP
#undef OUT
