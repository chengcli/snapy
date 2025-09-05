// torch
#include <ATen/Dispatch.h>
#include <ATen/TensorIterator.h>
#include <ATen/native/ReduceOpsUtils.h>
#include <c10/cuda/CUDAGuard.h>

// snap
#include <snap/loops.cuh>
#include "lmars_impl.h"
#include "hllc_impl.h"
#include "riemann_dispatch.hpp"

namespace snap {

void call_lmars_cuda(at::TensorIterator& iter, int dim) {
  at::cuda::CUDAGuard device_guard(iter.device());

  AT_DISPATCH_FLOATING_TYPES(iter.common_dtype(), "call_lmars_cuda", [&]() {
    auto nhydro = at::native::ensure_nonempty_size(iter.output(), 0);
    auto stride = at::native::ensure_nonempty_stride(iter.output(), 0);
    auto ny = nhydro - Index::ICY;

    native::gpu_kernel<5>(
        iter, [=] GPU_LAMBDA(char* const data[5], unsigned int strides[5]) {
          auto out = reinterpret_cast<scalar_t*>(data[0] + strides[0]);
          auto wl = reinterpret_cast<scalar_t*>(data[1] + strides[1]);
          auto wr = reinterpret_cast<scalar_t*>(data[2] + strides[2]);
          auto elr = reinterpret_cast<scalar_t*>(data[3] + strides[3]);
          auto glr = reinterpret_cast<scalar_t*>(data[4] + strides[4]);
          lmars_impl(out, wl, wr, *elr, *(elr + stride),
                     *glr, *(glr + stride), dim, ny, stride);
        });
  });
}

void call_hllc_cuda(at::TensorIterator& iter, int dim) {
  at::cuda::CUDAGuard device_guard(iter.device());

  AT_DISPATCH_FLOATING_TYPES(iter.common_dtype(), "call_hllc_cuda", [&]() {
    auto nhydro = at::native::ensure_nonempty_size(iter.output(), 0);
    auto stride = at::native::ensure_nonempty_stride(iter.output(), 0);
    auto ny = nhydro - Index::ICY;

    native::gpu_kernel<6>(
        iter, [=] GPU_LAMBDA(char* const data[6], unsigned int strides[6]) {
          auto out = reinterpret_cast<scalar_t*>(data[0] + strides[0]);
          auto wl = reinterpret_cast<scalar_t*>(data[1] + strides[1]);
          auto wr = reinterpret_cast<scalar_t*>(data[2] + strides[2]);
          auto elr = reinterpret_cast<scalar_t*>(data[3] + strides[3]);
          auto glr = reinterpret_cast<scalar_t*>(data[4] + strides[4]);
          auto clr = reinterpret_cast<scalar_t*>(data[5] + strides[5]);
          hllc_impl(out, wl, wr, *elr, *(elr + stride),
                    *glr, *(glr + stride), *clr, *(clr + stride),
                    dim, ny, stride);
        });
  });
}
}  // namespace snap

namespace at::native {

REGISTER_CUDA_DISPATCH(call_lmars, &snap::call_lmars_cuda);
REGISTER_CUDA_DISPATCH(call_hllc, &snap::call_hllc_cuda);

}  // namespace at::native
