// torch
#include <ATen/Dispatch.h>
#include <ATen/TensorIterator.h>
#include <ATen/native/ReduceOpsUtils.h>
#include <c10/cuda/CUDAGuard.h>

// snap
#include <snap/loops.cuh>
#include "flip_zero_impl.h"
#include "bc_dispatch.hpp"

namespace snap {

// device global number of flips
__device__ int d_num_flips = 0;

int flip_zero_cuda(at::TensorIterator& iter, int dim, int dir) {
  at::cuda::CUDAGuard device_guard(iter.device());

  int num_flips = 0;
  cudaMemcpyToSymbol(d_num_flips, &num_flips, sizeof(int));

  AT_DISPATCH_INTEGRAL_TYPES(iter.dtype(), "flip_zero_cuda", [&] {
    auto len = at::native::ensure_nonempty_size(iter.output(), dim);
    auto stride = at::native::ensure_nonempty_stride(iter.output(), dim);

    native::gpu_kernel<5>(
        iter, [=] __device__(char* const data[5], unsigned int strides[5]) {
          auto solid = reinterpret_cast<scalar_t*>(data[0] + strides[0]);
          auto dp = reinterpret_cast<scalar_t*>(data[1] + strides[1]);
          auto fromLen = reinterpret_cast<scalar_t*>(data[2] + strides[2]);
          auto fromBit = reinterpret_cast<scalar_t*>(data[3] + strides[3]);
          auto usedFlip = reinterpret_cast<scalar_t*>(data[4] + strides[4]);

          auto d = compute_min_flips(solid, len, /*minRun0=*/3, /*minRun1=*/2,
                                     /*allowBothFlips=*/0, dir, stride, dp,
                                     fromLen, fromBit, usedFlip);

          reconstruct_solution(solid, len, /*minRun0=*/3, /*minRun1=*/2,
                               /*allowBothFlips=*/0, dir, stride, fromLen,
                               fromBit, usedFlip);

          atomicAdd(&d_num_flips, d);
        });
  });

  cudaMemcpyFromSymbol(&num_flips, d_num_flips, sizeof(int));
  return num_flips;
}

}  // namespace snap

namespace at::native {

REGISTER_CUDA_DISPATCH(flip_zero, &snap::flip_zero_cuda);

}  // namespace at::native
