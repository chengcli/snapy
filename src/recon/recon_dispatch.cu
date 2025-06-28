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
  std::cout << "call poly == " << N << std::endl;

  AT_DISPATCH_FLOATING_TYPES(iter.common_dtype(), "call_poly_cuda", [&]() {
    int stride1 = at::native::ensure_nonempty_stride(iter.input(), dim);
    int stride2 = at::native::ensure_nonempty_stride(iter.input(), 0);

    int stride_out = at::native::ensure_nonempty_stride(iter.output(), 0);
    int nvar = at::native::ensure_nonempty_size(iter.output(), 0);
    int ndim = iter.output().dim();

    auto c = coeff.data_ptr<scalar_t>();

    native::stencil_kernel<scalar_t, 2>(
        iter, dim, 1, [=] GPU_LAMBDA(char* const data[2], unsigned int strides[2], scalar_t *smem) {
          auto out = reinterpret_cast<scalar_t*>(data[0] + strides[0]);
          auto w = reinterpret_cast<scalar_t*>(data[1] + strides[1]);
          interp_poly_impl<scalar_t, N>(out, w, c, dim, ndim, nvar,
                                        stride1, stride2, stride_out, smem);
        });
  });
}

void call_weno3_cuda(at::TensorIterator& iter, at::Tensor coeff, int dim, bool scale) {
  at::cuda::CUDAGuard device_guard(iter.device());

  /*AT_DISPATCH_FLOATING_TYPES(iter.common_dtype(), "call_weno3_cuda", [&]() {
    at::native::gpu_kernel(
        iter,
        [scale] GPU_LAMBDA(scalar_t in1, scalar_t in2,
                           scalar_t in3) -> scalar_t {
          auto s =
              scale ? (fabs(in1) + fabs(in2) + fabs(in3)) / 3. + 1.e-10 : 1.0;
          return s * interp_weno3(in1 / s, in2 / s, in3 / s);
        });
  });*/
}

void call_weno5_cuda(at::TensorIterator& iter, at::Tensor coeff, int dim, bool scale) {
  at::cuda::CUDAGuard device_guard(iter.device());

  /*AT_DISPATCH_FLOATING_TYPES(iter.common_dtype(), "call_weno5_cuda", [&]() {
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
  });*/
}
}  // namespace snap

namespace at::native {

REGISTER_CUDA_DISPATCH(call_poly3, &snap::call_poly_cuda<3>);
REGISTER_CUDA_DISPATCH(call_poly5, &snap::call_poly_cuda<5>);
REGISTER_CUDA_DISPATCH(call_weno3, &snap::call_weno3_cuda);
REGISTER_CUDA_DISPATCH(call_weno5, &snap::call_weno5_cuda);

}  // namespace at::native
