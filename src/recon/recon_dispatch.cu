// torch
#include <ATen/Dispatch.h>
#include <ATen/TensorIterator.h>
#include <c10/cuda/CUDAGuard.h>

// snap
#include <snap/loops.cuh>
#include "recon_dispatch.hpp"
#include "interp_impl.cuh"

namespace snap {

template <int N>
void call_poly_cuda(at::TensorIterator& iter, at::Tensor coeff, int dim) {
  at::cuda::CUDAGuard device_guard(iter.device());

  AT_DISPATCH_FLOATING_TYPES(iter.common_dtype(), "call_poly_cuda", [&]() {
    int stride_in1 = at::native::ensure_nonempty_stride(iter.input(), dim);
    int stride_in2 = at::native::ensure_nonempty_stride(iter.input(), 0);

    int stride_out1 = at::native::ensure_nonempty_stride(iter.output(), dim);
    int stride_out2 = at::native::ensure_nonempty_stride(iter.output(), 0);

    int nvar = at::native::ensure_nonempty_size(iter.output(), 0);
    auto c = coeff.data_ptr<scalar_t>();

    native::stencil_kernel<scalar_t, 2>(
        iter, dim, coeff.numel(),
        [=] GPU_LAMBDA(char* const data[2], unsigned int strides[2], scalar_t *smem) {
          auto out = reinterpret_cast<scalar_t*>(data[0] + strides[0]);
          auto w = reinterpret_cast<scalar_t*>(data[1] + strides[1]);
          interp_poly_impl<scalar_t, N>(out, w, c, nvar,
                                        stride_in1, stride_in2,
                                        stride_out1, stride_out2, smem);
        });
  });
}

void call_weno3_cuda(at::TensorIterator& iter, at::Tensor coeff, int dim, bool scale) {
  at::cuda::CUDAGuard device_guard(iter.device());

  AT_DISPATCH_FLOATING_TYPES(iter.common_dtype(), "call_weno3_cuda", [&]() {
    int stride_in1 = at::native::ensure_nonempty_stride(iter.input(), dim);
    int stride_in2 = at::native::ensure_nonempty_stride(iter.input(), 0);

    int stride_out1 = at::native::ensure_nonempty_stride(iter.output(), dim);
    int stride_out2 = at::native::ensure_nonempty_stride(iter.output(), 0);

    int nvar = at::native::ensure_nonempty_size(iter.output(), 0);
    auto c = coeff.data_ptr<scalar_t>();

    native::stencil_kernel<scalar_t, 2>(
        iter, dim, coeff.numel(),
        [=] __device__ (char* const data[2], unsigned int strides[2], scalar_t *smem) {
          auto out = reinterpret_cast<scalar_t*>(data[0] + strides[0]);
          auto w = reinterpret_cast<scalar_t*>(data[1] + strides[1]);
          interp_weno3_impl(out, w, c, nvar,
                            stride_in1, stride_in2,
                            stride_out1, stride_out2, scale, smem);
        });
  });
}

void call_weno5_cuda(at::TensorIterator& iter, at::Tensor coeff, int dim, bool scale) {
  at::cuda::CUDAGuard device_guard(iter.device());

  AT_DISPATCH_FLOATING_TYPES(iter.common_dtype(), "call_weno5_cuda", [&]() {
    int stride_in1 = at::native::ensure_nonempty_stride(iter.input(), dim);
    int stride_in2 = at::native::ensure_nonempty_stride(iter.input(), 0);

    int stride_out1 = at::native::ensure_nonempty_stride(iter.output(), dim);
    int stride_out2 = at::native::ensure_nonempty_stride(iter.output(), 0);

    int nvar = at::native::ensure_nonempty_size(iter.output(), 0);
    auto c = coeff.data_ptr<scalar_t>();

    native::stencil_kernel<scalar_t, 2>(
        iter, dim, coeff.numel(),
        [=] __device__ (char* const data[2], unsigned int strides[2], scalar_t *smem) {
          auto out = reinterpret_cast<scalar_t*>(data[0] + strides[0]);
          auto w = reinterpret_cast<scalar_t*>(data[1] + strides[1]);
          interp_weno5_impl(out, w, c, nvar,
                            stride_in1, stride_in2,
                            stride_out1, stride_out2, scale, smem);
        });
  });
}
}  // namespace snap

namespace at::native {

REGISTER_CUDA_DISPATCH(call_poly3, &snap::call_poly_cuda<3>);
REGISTER_CUDA_DISPATCH(call_poly5, &snap::call_poly_cuda<5>);
REGISTER_CUDA_DISPATCH(call_weno3, &snap::call_weno3_cuda);
REGISTER_CUDA_DISPATCH(call_weno5, &snap::call_weno5_cuda);

}  // namespace at::native
