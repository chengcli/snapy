// torch
#include <ATen/Dispatch.h>
#include <ATen/TensorIterator.h>
#include <ATen/native/ReduceOpsUtils.h>
#include <c10/cuda/CUDAGuard.h>

// snap
#include <snap/utils/loops.cuh>
#include "coord_dispatch.hpp"
#include "coord_utils_impl.h"

namespace snap {

void call_coord_vec_lower_cuda(at::TensorIterator& iter) {
  at::cuda::CUDAGuard device_guard(iter.device());

  AT_DISPATCH_FLOATING_TYPES(iter.common_dtype(), "call_coord_vec_lower_cuda", [&]() {
    native::gpu_kernel<3>(
        iter, [=] GPU_LAMBDA(char* const data[3], unsigned int strides[3]) {
          auto v2 = reinterpret_cast<scalar_t*>(data[0] + strides[0]);
          auto v3 = reinterpret_cast<scalar_t*>(data[1] + strides[1]);
          auto cth = reinterpret_cast<scalar_t*>(data[2] + strides[2]);
          coord_vec_lower_impl(v2, v3, *cth);
        });
  });
}

void call_coord_vec_raise_cuda(at::TensorIterator& iter) {
  at::cuda::CUDAGuard device_guard(iter.device());

  AT_DISPATCH_FLOATING_TYPES(iter.common_dtype(), "call_coord_vec_raise_cuda", [&]() {
    native::gpu_kernel<3>(
        iter, [=] GPU_LAMBDA(char* const data[3], unsigned int strides[3]) {
          auto v2 = reinterpret_cast<scalar_t*>(data[0] + strides[0]);
          auto v3 = reinterpret_cast<scalar_t*>(data[1] + strides[1]);
          auto cth = reinterpret_cast<scalar_t*>(data[2] + strides[2]);
          coord_vec_raise_impl(v2, v3, *cth);
        });
  });
}

} // namespace snap

namespace at::native {

REGISTER_CUDA_DISPATCH(call_coord_vec_lower,
                       &snap::call_coord_vec_lower_cuda);
REGISTER_CUDA_DISPATCH(call_coord_vec_raise,
                       &snap::call_coord_vec_raise_cuda);

}  // namespace at::native
