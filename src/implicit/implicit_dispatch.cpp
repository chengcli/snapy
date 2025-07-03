// eigen
#include <Eigen/Dense>

// torch
#include <ATen/Dispatch.h>
#include <ATen/TensorIterator.h>
#include <ATen/native/ReduceOpsUtils.h>

// snap
#include "implicit_dispatch.hpp"
#include "tridiag_thomas_impl.h"

namespace snap {

template <int N>
void alloc_eigen_cpu(char *&a, char *&b, char *&c, char *&delta, char *&corr,
                     int ncol, int nlayer, c10::ScalarType dtype) {
  AT_DISPATCH_FLOATING_TYPES(dtype, "alloc_eigen_cpu", [&] {
    a = reinterpret_cast<char *>(
        new Eigen::Matrix<scalar_t, N, N, Eigen::RowMajor>[ncol * nlayer]);
    b = reinterpret_cast<char *>(
        new Eigen::Matrix<scalar_t, N, N, Eigen::RowMajor>[ncol * nlayer]);
    c = reinterpret_cast<char *>(
        new Eigen::Matrix<scalar_t, N, N, Eigen::RowMajor>[ncol * nlayer]);
    delta =
        reinterpret_cast<char *>(new Eigen::Vector<scalar_t, N>[ncol * nlayer]);
    corr =
        reinterpret_cast<char *>(new Eigen::Vector<scalar_t, N>[ncol * nlayer]);
  });
}

template <int N>
void vic_forward_cpu(at::TensorIterator &iter, double dt, int il, int iu) {
  AT_DISPATCH_FLOATING_TYPES(iter.dtype(), "vic_forward_cpu", [&] {
    auto nhydro = at::native::ensure_nonempty_size(iter.output(), 0);
    auto stride = at::native::ensure_nonempty_stride(iter.output(), 0);

    iter.for_each([&](char **data, const int64_t *strides, int64_t n) {
      for (int i = 0; i < n; i++) {
        auto du = reinterpret_cast<scalar_t *>(data[0] + i * strides[0]);
        auto w = reinterpret_cast<scalar_t *>(data[1] + i * strides[1]);
        auto a =
            reinterpret_cast<Eigen::Matrix<scalar_t, N, N, Eigen::RowMajor> *>(
                data[2] + i * strides[2]);
        auto b =
            reinterpret_cast<Eigen::Matrix<scalar_t, N, N, Eigen::RowMajor> *>(
                data[3] + i * strides[3]);
        auto c =
            reinterpret_cast<Eigen::Matrix<scalar_t, N, N, Eigen::RowMajor> *>(
                data[4] + i * strides[4]);
        auto delta = reinterpret_cast<Eigen::Vector<scalar_t, N> *>(
            data[5] + i * strides[5]);
        auto corr = reinterpret_cast<Eigen::Vector<scalar_t, N> *>(
            data[6] + i * strides[6]);

        forward_sweep_impl(a, b, c, delta, corr, du, dt, nhydro, stride, il,
                           iu);
        backward_substitution_impl(a, delta, w, du, nhydro, stride, il, iu);
      }
    });
  });
}

void free_eigen_cpu(char *&a, char *&b, char *&c, char *&delta, char *&corr) {
  delete[] a;
  delete[] b;
  delete[] c;
  delete[] delta;
  delete[] corr;
}

}  // namespace snap

namespace at::native {

DEFINE_DISPATCH(vic_forward);

REGISTER_ALL_CPU_DISPATCH(vic_forward3, &snap::vic_forward_cpu<3>);
REGISTER_ALL_CPU_DISPATCH(vic_forward3, &snap::vic_forward_cpu<5>);

REGISTER_ALL_CPU_DISPATCH(alloc_eigen3, &snap::alloc_eigen_cpu<3>);
REGISTER_ALL_CPU_DISPATCH(alloc_eigen5, &snap::alloc_eigen_cpu<5>);

REGISTER_ALL_CPU_DISPATCH(free_eigen, &snap::free_eigen_cpu);

}  // namespace at::native
