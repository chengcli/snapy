// eigen
#include <Eigen/Dense>

// torch
#include <ATen/Dispatch.h>
#include <ATen/TensorIterator.h>
#include <ATen/native/ReduceOpsUtils.h>

// fvm
#include "tridiag_thomas_impl.h"

namespace snap {
template <int N>
void vic_forward_cpu(at::TensorIterator& iter, double dt, int il, int iu) {
  AT_DISPATCH_FLOATING_TYPES(iter.dtype(), "vic_forward_cpu", [&] {
    auto nhydro = at::native::ensure_nonempty_size(iter.output(), 0);
    auto stride = at::native::ensure_nonempty_stride(iter.output(), 0);

    iter.for_each([&](char** data, const int64_t* strides, int64_t n) {
      for (int i = 0; i < n; i++) {
        auto du = reinterpret_cast<scalar_t*>(data[0] + i * strides[0]);
        auto w = reinterpret_cast<scalar_t*>(data[1] + i * strides[1]);
        auto a =
            reinterpret_cast<Eigen::Matrix<scalar_t, N, N, Eigen::RowMajor>*>(
                data[2] + i * strides[2]);
        auto b =
            reinterpret_cast<Eigen::Matrix<scalar_t, N, N, Eigen::RowMajor>*>(
                data[3] + i * strides[3]);
        auto c =
            reinterpret_cast<Eigen::Matrix<scalar_t, N, N, Eigen::RowMajor>*>(
                data[4] + i * strides[4]);
        auto delta = reinterpret_cast<Eigen::Vector<scalar_t, N>*>(
            data[5] + i * strides[5]);
        auto corr = reinterpret_cast<Eigen::Vector<scalar_t, N>*>(
            data[6] + i * strides[6]);

        forward_sweep_impl(a, b, c, delta, corr, du, dt, nhydro, stride, il,
                           iu);
        backward_substitution_impl(a, delta, w, du, nhydro, stride, il, iu);
      }
    });
  });
}

template void vic_forward_cpu<3>(at::TensorIterator& iter, double dt, int il,
                                 int iu);
template void vic_forward_cpu<5>(at::TensorIterator& iter, double dt, int il,
                                 int iu);

}  // namespace snap
