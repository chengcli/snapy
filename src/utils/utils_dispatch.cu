// torch
#include <ATen/Dispatch.h>
#include <ATen/TensorIterator.h>
#include <ATen/native/ReduceOpsUtils.h>
#include <c10/cuda/CUDAGuard.h>

// snap
#include <snap/loops.cuh>
#include "utils_dispatch.hpp"

#define INP1(j, i) (inp1[(j) * stride_in2 + (i) * stride_in1])
#define INP2(j, i) (inp2[(j) * stride_in2 + (i) * stride_in1])
#define OUT(j, i) (out[(j) * stride_out2 + (i) * stride_out1])

namespace snap {

template <typename T>
__device__ void bdot_out_impl(T *out, T *inp1, T *inp2,
                              int nvar, int stride_in1, int stride_in2,
                              int stride_out1, int stride_out2, float scale, T *smem) {
  int id = threadIdx.x;
  int nt = blockDim.x;

  // each thread multiplies one element
  for (int j = 0; j < nvar; ++j) {
    smem[id + j * nt] = INP1(j, id) * INP2(j, id);
  }

  __syncthreads();

  // tree‐based reduction in shared memory
  for (unsigned int s = nt/2; s > 0; s >>= 1) {
    if (id < s) {
      for (int j = 0; j < nvar; ++j)
        smem[id + j * nt] += smem[id + s + j * nt];
    }
    __syncthreads();
  }

  // write to global memory
  for (int j = id; j < nvar; j += nt) {
    OUT(j, id) = smem[j * nt] * scale;
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
    int stride_in1 = at::native::ensure_nonempty_stride(iter.input(), dim);
    int stride_in2 = at::native::ensure_nonempty_stride(iter.input(), 0);

    int stride_out1 = at::native::ensure_nonempty_stride(iter.output(), dim);
    int stride_out2 = at::native::ensure_nonempty_stride(iter.output(), 0);

    int nvar = at::native::ensure_nonempty_size(iter.output(), 0);

    native::stencil_kernel<scalar_t, 3>(
        iter, dim, 0,
        [=] __device__ (char* const data[3], unsigned int strides[3], scalar_t *smem) {
          auto out = reinterpret_cast<scalar_t*>(data[0] + strides[0]);
          auto inp1 = reinterpret_cast<scalar_t*>(data[1] + strides[1]);
          auto inp2 = reinterpret_cast<scalar_t*>(data[2] + strides[2]);
          bdot_out_impl<scalar_t>(out, inp1, inp2, dim,
                                  stride_in1, stride_in2, stride_out1, stride_out2,
                                  scale, smem);
        });
  });
}

} // namespace snap

namespace at::native {

REGISTER_CUDA_DISPATCH(bdot_out, &snap::bdot_out_cuda);

}  // namespace at::native

#undef INP
#undef OUT
