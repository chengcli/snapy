// eigen
#include <Eigen/Dense>

// torch
#include <ATen/Dispatch.h>
#include <ATen/TensorIterator.h>
#include <ATen/native/ReduceOpsUtils.h>
#include <c10/cuda/CUDAGuard.h>

// snap
#include <snap/loops.cuh>
#include "tridiag_thomas_impl.h"

namespace snap {
template <int N>
void vic_forward_cuda(at::TensorIterator& iter, double dt, int il, int iu) {
  at::cuda::CUDAGuard device_guard(iter.device());

  AT_DISPATCH_FLOATING_TYPES(iter.dtype(), "vic_forward_cuda", [&]() {
    auto nhydro = at::native::ensure_nonempty_size(iter.output(), 0);
    auto stride = at::native::ensure_nonempty_stride(iter.output(), 0);

    native::gpu_kernel<7>(iter, [=] GPU_LAMBDA(
                                              char* const data[7],
                                              unsigned int strides[7]) {
      auto du = reinterpret_cast<scalar_t*>(data[0] + strides[0]);
      auto w = reinterpret_cast<scalar_t*>(data[1] + strides[1]);
      auto a =
          reinterpret_cast<Eigen::Matrix<scalar_t, N, N, Eigen::RowMajor>*>(
              data[2] + strides[2]);
      auto b =
          reinterpret_cast<Eigen::Matrix<scalar_t, N, N, Eigen::RowMajor>*>(
              data[3] + strides[3]);
      auto c =
          reinterpret_cast<Eigen::Matrix<scalar_t, N, N, Eigen::RowMajor>*>(
              data[4] + strides[4]);
      auto delta =
          reinterpret_cast<Eigen::Vector<scalar_t, N>*>(data[5] + strides[5]);
      auto corr =
          reinterpret_cast<Eigen::Vector<scalar_t, N>*>(data[6] + strides[6]);

      forward_sweep_impl(a, b, c, delta, corr, du, dt, nhydro, stride, il, iu);
      backward_substitution_impl(a, delta, w, du, nhydro, stride, il, iu);
    });
  });
}

template void vic_forward_cuda<3>(at::TensorIterator& iter, double dt, int il,
                                  int iu);
template void vic_forward_cuda<5>(at::TensorIterator& iter, double dt, int il,
                                  int iu);
}  // namespace snap
