// torch
#include <ATen/Dispatch.h>
#include <ATen/TensorIterator.h>
#include <ATen/native/ReduceOpsUtils.h>
#include <c10/cuda/CUDAGuard.h>

// fmv
#include <snap/loops.cuh>
#include "lmars_impl.h"
#include "hllc_impl.h"

namespace snap {
void call_lmars_cuda(at::TensorIterator& iter, int dim, int nvapor) {
  at::cuda::CUDAGuard device_guard(iter.device());

  AT_DISPATCH_FLOATING_TYPES(iter.common_dtype(), "lmars_cuda", [&]() {
    auto nhydro = at::native::ensure_nonempty_size(iter.output(), 0);
    auto stride = at::native::ensure_nonempty_stride(iter.output(), 0);
    auto ncloud = nhydro - nvapor - Index::ICY;

    native::gpu_kernel<scalar_t, 6>(
        iter, [=] GPU_LAMBDA(char* const data[6], unsigned int strides[6]) {
          auto out = reinterpret_cast<scalar_t*>(data[0] + strides[0]);
          auto wli = reinterpret_cast<scalar_t*>(data[1] + strides[1]);
          auto wri = reinterpret_cast<scalar_t*>(data[2] + strides[2]);
          auto gammad = reinterpret_cast<scalar_t*>(data[3] + strides[3]);
          auto cv_ratio_m1 = reinterpret_cast<scalar_t*>(data[4] + strides[4]);
          auto mu_ratio_m1 = reinterpret_cast<scalar_t*>(data[5] + strides[5]);
          lmars_impl(out, wli, wri, gammad, cv_ratio_m1, mu_ratio_m1, dim,
                     nvapor, ncloud, stride);
        });
  });
}

void call_hllc_cuda(at::TensorIterator& iter, int dim, int nvapor) {
  at::cuda::CUDAGuard device_guard(iter.device());

  AT_DISPATCH_FLOATING_TYPES(iter.common_dtype(), "hllc_cuda", [&]() {
    auto nhydro = at::native::ensure_nonempty_size(iter.output(), 0);
    auto stride = at::native::ensure_nonempty_stride(iter.output(), 0);
    auto ncloud = nhydro - nvapor - Index::ICY;

    native::gpu_kernel<scalar_t, 6>(
        iter, [=] GPU_LAMBDA(char* const data[6], unsigned int strides[6]) {
          auto out = reinterpret_cast<scalar_t*>(data[0] + strides[0]);
          auto wli = reinterpret_cast<scalar_t*>(data[1] + strides[1]);
          auto wri = reinterpret_cast<scalar_t*>(data[2] + strides[2]);
          auto gammad = reinterpret_cast<scalar_t*>(data[3] + strides[3]);
          auto cv_ratio_m1 = reinterpret_cast<scalar_t*>(data[4] + strides[4]);
          auto mu_ratio_m1 = reinterpret_cast<scalar_t*>(data[5] + strides[5]);
          hllc_impl(out, wli, wri, gammad, cv_ratio_m1, mu_ratio_m1, dim,
                    nvapor, ncloud, stride);
        });
  });
}
}  // namespace snap
