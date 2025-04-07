// torch
#include <ATen/Dispatch.h>
#include <ATen/TensorIterator.h>
#include <ATen/native/cuda/Loops.cuh>
#include <c10/cuda/CUDAGuard.h>

// fvm
#include "interp_simple.hpp"

namespace snap {
void call_cp3_cuda(at::TensorIterator& iter) {
  at::cuda::CUDAGuard device_guard(iter.device());

  AT_DISPATCH_FLOATING_TYPES(iter.common_dtype(), "cp3_cuda", [&]() {
    at::native::gpu_kernel(
        iter,
        [] GPU_LAMBDA(scalar_t in1, scalar_t in2, scalar_t in3) -> scalar_t {
          return interp_cp3(in1, in2, in3);
        });
  });
}

void call_cp5_cuda(at::TensorIterator& iter) {
  at::cuda::CUDAGuard device_guard(iter.device());

  AT_DISPATCH_FLOATING_TYPES(iter.common_dtype(), "cp5_cuda", [&]() {
    at::native::gpu_kernel(
        iter,
        [] GPU_LAMBDA(scalar_t in1, scalar_t in2, scalar_t in3, scalar_t in4,
                      scalar_t in5) -> scalar_t {
          return interp_cp5(in1, in2, in3, in4, in5);
        });
  });
}

void call_weno3_cuda(at::TensorIterator& iter, bool scale) {
  at::cuda::CUDAGuard device_guard(iter.device());

  AT_DISPATCH_FLOATING_TYPES(iter.common_dtype(), "weno3_cuda", [&]() {
    at::native::gpu_kernel(
        iter,
        [scale] GPU_LAMBDA(scalar_t in1, scalar_t in2,
                           scalar_t in3) -> scalar_t {
          auto s =
              scale ? (fabs(in1) + fabs(in2) + fabs(in3)) / 3. + 1.e-10 : 1.0;
          return s * interp_weno3(in1 / s, in2 / s, in3 / s);
        });
  });
}

void call_weno5_cuda(at::TensorIterator& iter, bool scale) {
  at::cuda::CUDAGuard device_guard(iter.device());

  AT_DISPATCH_FLOATING_TYPES(iter.common_dtype(), "weno5_cuda", [&]() {
    at::native::gpu_kernel(
        iter,
        [scale] GPU_LAMBDA(scalar_t in1, scalar_t in2, scalar_t in3,
                           scalar_t in4, scalar_t in5) -> scalar_t {
          auto s = scale ? (fabs(in1) + fabs(in2) + fabs(in3) + fabs(in4) +
                            fabs(in5)) /
                                   5. +
                               1.e-10
                         : 1.0;
          return s * interp_weno5(in1 / s, in2 / s, in3 / s, in4 / s, in5 / s);
        });
  });
}
}  // namespace snap
