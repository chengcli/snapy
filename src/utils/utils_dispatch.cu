// torch
#include <ATen/Dispatch.h>
#include <ATen/TensorIterator.h>
#include <ATen/native/ReduceOpsUtils.h>
#include <c10/cuda/CUDAGuard.h>

// snap
#include <snap/loops.cuh>
#include "utils_dispatch.hpp"

#define INP1(j, i) (inp1[(j) * stride2 + (i) * stride1])
#define INP2(j, i) (inp2[(j) * stride2 + (i) * stride1])
#define OUT(j) (out[(j) * stride_out])

namespace snap {

template <typename T>
__device__ void bdot_out_impl(T *out, T *inp1, T *inp2, int dim, int ndim,
                              int nvar, int stride1, int stride2,
                              int stride_out, float scale, T *smem) {
  unsigned int idx[3] = {threadIdx.z, threadIdx.y, threadIdx.x};
  unsigned int len[3] = {blockDim.z, blockDim.y, blockDim.x};

  int idim = 3 + dim - ndim;
  int tid = idx[idim];

  // each thread multiplies one element
  for (int j = 0; j < nvar; ++j) {
    smem[tid + j * len[idim]] = INP1(j, tid) * INP2(j, tid);
  }

  __syncthreads();

  // tree‐based reduction in shared memory
  for (unsigned int s = len[idim]/2; s > 0; s >>= 1) {
    if (tid < s) {
      for (int j = 0; j < nvar; ++j)
        smem[tid + j * len[idim]] += smem[tid + s + j * len[idim]];
    }
    __syncthreads();
  }

  // write to global memory
  for (int j = tid; j < nvar; j += len[idim]) {
    OUT(j) = smem[j * len[idim]] * scale;
  }
}

void bdot_out_cuda(
    at::Tensor &out, at::Tensor const &inp1, at::Tensor const &inp2,
    float scale, int dim) {

  auto iter = at::TensorIteratorConfig()
      .resize_outputs(false)
      .check_all_same_dtype(false)
      .declare_static_shape(out.sizes(), /*squash_dim=*/{0})
      .add_output(out)
      .add_input(inp1)
      .add_input(inp2)
      .build();

  at::cuda::CUDAGuard device_guard(iter.device());

  AT_DISPATCH_FLOATING_TYPES(iter.common_dtype(), "bdot_out_cuda", [&]() {
    int stride1 = at::native::ensure_nonempty_stride(iter.input(), dim);
    int stride2 = at::native::ensure_nonempty_stride(iter.input(), 0);

    int stride_out = at::native::ensure_nonempty_stride(iter.output(), 0);
    int nvar = at::native::ensure_nonempty_size(iter.output(), 0);
    int ndim = iter.output().dim();

    native::stencil_kernel<scalar_t, 3>(
        iter, dim, 0,
        [=] __device__ (char* const data[3], unsigned int strides[3], scalar_t *smem) {
          auto out = reinterpret_cast<scalar_t*>(data[0] + strides[0]);
          auto inp1 = reinterpret_cast<scalar_t*>(data[1] + strides[1]);
          auto inp2 = reinterpret_cast<scalar_t*>(data[2] + strides[2]);
          bdot_out_impl<scalar_t>(out, inp1, inp2, dim, ndim, nvar,
                                  stride1, stride2, stride_out, scale, smem);
        });
  });
}

} // namespace snap

namespace at::native {

REGISTER_CUDA_DISPATCH(bdot_out, &snap::bdot_out_cuda);

}  // namespace at::native

#undef INP
#undef OUT
