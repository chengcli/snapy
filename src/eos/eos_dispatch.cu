// torch
#include <ATen/Dispatch.h>
#include <ATen/TensorIterator.h>
#include <ATen/native/ReduceOpsUtils.h>
#include <c10/cuda/CUDAGuard.h>

// fmv
#include <fvm/loops.cuh>
#include "ideal_gas_impl.h"
#include "ideal_moist_impl.h"

namespace snap {

void call_ideal_gas_cuda(at::TensorIterator& iter) {
  at::cuda::CUDAGuard device_guard(iter.device());

  AT_DISPATCH_FLOATING_TYPES(iter.common_dtype(), "ideal_gas_cuda", [&]() {
    auto stride = at::native::ensure_nonempty_stride(iter.output(), 0);

    native::gpu_kernel<scalar_t, 3>(
        iter, [=] GPU_LAMBDA(char* const data[3], unsigned int strides[3]) {
          auto prim = reinterpret_cast<scalar_t*>(data[0] + strides[0]);
          auto cons = reinterpret_cast<scalar_t*>(data[1] + strides[1]);
          auto gammad = reinterpret_cast<scalar_t*>(data[2] + strides[2]);
          ideal_gas_cons2prim(prim, cons, gammad, stride);
        });
  });
}

void call_ideal_moist_cuda(at::TensorIterator& iter) {
  at::cuda::CUDAGuard device_guard(iter.device());

  AT_DISPATCH_FLOATING_TYPES(iter.common_dtype(), "ideal_moist_cuda", [&]() {
    auto stride = at::native::ensure_nonempty_stride(iter.output(), 0);
    auto nhydro = at::native::ensure_nonempty_size(iter.output(), 0);
    auto nmass = nhydro - 5;

    native::gpu_kernel<scalar_t, 5>(
        iter, [=] GPU_LAMBDA(char* const data[5], unsigned int strides[5]) {
          auto prim = reinterpret_cast<scalar_t*>(data[0] + strides[0]);
          auto cons = reinterpret_cast<scalar_t*>(data[1] + strides[1]);
          auto gammad = reinterpret_cast<scalar_t*>(data[2] + strides[2]);
          auto feps = reinterpret_cast<scalar_t*>(data[3] + strides[3]);
          auto fsig = reinterpret_cast<scalar_t*>(data[4] + strides[4]);
          ideal_moist_cons2prim(prim, cons, gammad, feps, fsig, nmass, stride);
        });
  });
}

}  // namespace snap
