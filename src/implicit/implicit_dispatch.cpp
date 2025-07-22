// eigen
#include <Eigen/Dense>

// torch
#include <ATen/Dispatch.h>
#include <ATen/TensorIterator.h>
#include <ATen/native/ReduceOpsUtils.h>
#include <torch/torch.h>

// snap
#include "flux_decomposition_impl.h"
#include "implicit_dispatch.hpp"
#include "tridiag_thomas_impl.h"

namespace snap {

void call_roe_average_cpu(at::TensorIterator &iter) {
  int grain_size = iter.numel() / at::get_num_threads();

  AT_DISPATCH_FLOATING_TYPES(iter.dtype(), "call_roe_average_cpu", [&] {
    auto stride = at::native::ensure_nonempty_stride(iter.output(), 0);

    iter.for_each(
        [&](char **data, const int64_t *strides, int64_t n) {
          for (int i = 0; i < n; i++) {
            auto wroe = reinterpret_cast<scalar_t *>(data[0] + i * strides[0]);
            auto wl = reinterpret_cast<scalar_t *>(data[1] + i * strides[1]);
            auto wr = reinterpret_cast<scalar_t *>(data[2] + i * strides[2]);
            auto gamma = reinterpret_cast<scalar_t *>(data[3] + i * strides[3]);
            roe_average_impl(wroe, wl, wr, *gamma, stride);
          }
        },
        grain_size);
  });
}

void call_eigen_system_cpu(at::TensorIterator &iter, int dim) {
  int grain_size = iter.numel() / at::get_num_threads();

  AT_DISPATCH_FLOATING_TYPES(iter.dtype(), "call_eigen_system_cpu", [&] {
    auto stride = at::native::ensure_nonempty_stride(iter.input(), 0);

    iter.for_each(
        [&](char **data, const int64_t *strides, int64_t n) {
          for (int i = 0; i < n; i++) {
            auto Rmat = reinterpret_cast<scalar_t *>(data[0] + i * strides[0]);
            auto Rimat = reinterpret_cast<scalar_t *>(data[1] + i * strides[1]);
            auto EV = reinterpret_cast<scalar_t *>(data[2] + i * strides[2]);
            auto wroe = reinterpret_cast<scalar_t *>(data[3] + i * strides[3]);
            auto gamma = reinterpret_cast<scalar_t *>(data[4] + i * strides[4]);
            eigen_system_impl(Rmat, Rimat, EV, wroe, *gamma, dim, stride);
          }
        },
        grain_size);
  });
}

void call_flux_jacobian_cpu(at::TensorIterator &iter, int dim) {
  int grain_size = iter.numel() / at::get_num_threads();

  AT_DISPATCH_FLOATING_TYPES(iter.dtype(), "call_flux_jacobian_cpu", [&] {
    auto stride = at::native::ensure_nonempty_stride(iter.input(), 0);

    iter.for_each(
        [&](char **data, const int64_t *strides, int64_t n) {
          for (int i = 0; i < n; i++) {
            auto dfdq = reinterpret_cast<scalar_t *>(data[0] + i * strides[0]);
            auto wroe = reinterpret_cast<scalar_t *>(data[1] + i * strides[1]);
            auto gamma = reinterpret_cast<scalar_t *>(data[2] + i * strides[2]);
            flux_jacobian_impl(dfdq, wroe, *gamma, dim, stride);
          }
        },
        grain_size);
  });
}

template <int N>
void vic_solve_cpu(at::TensorIterator &iter, double dt, int il, int iu) {
  int grain_size = iter.numel() / at::get_num_threads();

  AT_DISPATCH_FLOATING_TYPES(iter.dtype(), "vic_solve_cpu", [&] {
    auto nhydro = at::native::ensure_nonempty_size(iter.output(), 0);
    auto stride = at::native::ensure_nonempty_stride(iter.output(), 0);
    auto ny = nhydro - Index::ICY;

    iter.for_each(
        [&](char **data, const int64_t *strides, int64_t n) {
          for (int i = 0; i < n; i++) {
            auto du = reinterpret_cast<scalar_t *>(data[0] + i * strides[0]);
            auto w = reinterpret_cast<scalar_t *>(data[1] + i * strides[1]);
            auto a = reinterpret_cast<
                Eigen::Matrix<scalar_t, N, N, Eigen::RowMajor> *>(
                data[2] + i * strides[2]);
            auto b = reinterpret_cast<
                Eigen::Matrix<scalar_t, N, N, Eigen::RowMajor> *>(
                data[3] + i * strides[3]);
            auto c = reinterpret_cast<
                Eigen::Matrix<scalar_t, N, N, Eigen::RowMajor> *>(
                data[4] + i * strides[4]);
            auto delta = reinterpret_cast<Eigen::Vector<scalar_t, N> *>(
                data[5] + i * strides[5]);
            auto corr = reinterpret_cast<Eigen::Vector<scalar_t, N> *>(
                data[6] + i * strides[6]);

            forward_sweep_impl(a, b, c, delta, corr, du, dt, ny, stride, il,
                               iu);
            backward_substitution_impl(a, delta, w, du, ny, stride, il, iu);
          }
        },
        grain_size);
  });
}

template <int N>
void alloc_eigen_cpu(c10::ScalarType dtype, char *&a, char *&b, char *&c,
                     char *&delta, char *&corr, int ncol, int nlayer) {
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

void free_eigen_cpu(char *&a, char *&b, char *&c, char *&delta, char *&corr) {
  delete[] a;
  delete[] b;
  delete[] c;
  delete[] delta;
  delete[] corr;
}

}  // namespace snap

namespace at::native {

DEFINE_DISPATCH(call_roe_average);
DEFINE_DISPATCH(call_eigen_system);
DEFINE_DISPATCH(call_flux_jacobian);

DEFINE_DISPATCH(vic_solve3);
DEFINE_DISPATCH(vic_solve5);

DEFINE_DISPATCH(alloc_eigen3);
DEFINE_DISPATCH(alloc_eigen5);
DEFINE_DISPATCH(free_eigen);

REGISTER_ALL_CPU_DISPATCH(call_roe_average, &snap::call_roe_average_cpu);
REGISTER_ALL_CPU_DISPATCH(call_eigen_system, &snap::call_eigen_system_cpu);
REGISTER_ALL_CPU_DISPATCH(call_flux_jacobian, &snap::call_flux_jacobian_cpu);

REGISTER_ALL_CPU_DISPATCH(vic_solve3, &snap::vic_solve_cpu<3>);
REGISTER_ALL_CPU_DISPATCH(vic_solve5, &snap::vic_solve_cpu<5>);

REGISTER_ALL_CPU_DISPATCH(alloc_eigen3, &snap::alloc_eigen_cpu<3>);
REGISTER_ALL_CPU_DISPATCH(alloc_eigen5, &snap::alloc_eigen_cpu<5>);
REGISTER_ALL_CPU_DISPATCH(free_eigen, &snap::free_eigen_cpu);

}  // namespace at::native
