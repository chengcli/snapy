// eigen
#include <Eigen/Dense>

// torch
#include <ATen/Dispatch.h>
#include <ATen/TensorIterator.h>
#include <ATen/native/ReduceOpsUtils.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/core/ScalarType.h>

// snap
#include <snap/utils/cuda_utils.h>
#include <snap/loops.cuh>
#include "flux_decomposition_impl.h"
#include "tridiag_thomas_impl.h"
#include "implicit_dispatch.hpp"

namespace snap {

void call_roe_average_cuda(at::TensorIterator &iter) {
  at::cuda::CUDAGuard device_guard(iter.device());

  AT_DISPATCH_FLOATING_TYPES(iter.dtype(), "call_roe_average_cuda", [&] {
    auto stride = at::native::ensure_nonempty_stride(iter.output(), 0);

    native::gpu_kernel<4>(iter, [=] GPU_LAMBDA(
        char * const data[4], unsigned int strides[4]) {
          auto wroe = reinterpret_cast<scalar_t *>(data[0] + strides[0]);
          auto wl = reinterpret_cast<scalar_t *>(data[1] + strides[1]);
          auto wr = reinterpret_cast<scalar_t *>(data[2] + strides[2]);
          auto gamma = reinterpret_cast<scalar_t *>(data[3] + strides[3]);
          roe_average_impl(wroe, wl, wr, *gamma, stride);
        });
  });
}

void call_eigen_system_cuda(at::TensorIterator &iter, int dim) {
  at::cuda::CUDAGuard device_guard(iter.device());

  AT_DISPATCH_FLOATING_TYPES(iter.dtype(), "call_eigen_system_cuda", [&] {
    auto stride = at::native::ensure_nonempty_stride(iter.input(), 0);

    native::gpu_kernel<5>(iter, [=] GPU_LAMBDA(
        char * const data[5], unsigned int strides[5]) {
          auto Rmat = reinterpret_cast<scalar_t *>(data[0] + strides[0]);
          auto Rimat = reinterpret_cast<scalar_t *>(data[1] + strides[1]);
          auto EV = reinterpret_cast<scalar_t *>(data[2] + strides[2]);
          auto wroe = reinterpret_cast<scalar_t *>(data[3] + strides[3]);
          auto gamma = reinterpret_cast<scalar_t *>(data[4] + strides[4]);
          eigen_system_impl(Rmat, Rimat, EV, wroe, *gamma, dim, stride);
        });
  });
}

void call_flux_jacobian_cuda(at::TensorIterator &iter, int dim) {
  at::cuda::CUDAGuard device_guard(iter.device());

  AT_DISPATCH_FLOATING_TYPES(iter.dtype(), "call_flux_jacobian_cuda", [&] {
    auto stride = at::native::ensure_nonempty_stride(iter.input(), 0);

    native::gpu_kernel<3>(iter, [=] GPU_LAMBDA(
        char * const data[3], unsigned int strides[3]) {
          auto dfdq = reinterpret_cast<scalar_t *>(data[0] + strides[0]);
          auto wroe = reinterpret_cast<scalar_t *>(data[1] + strides[1]);
          auto gamma = reinterpret_cast<scalar_t *>(data[2] + strides[2]);
          flux_jacobian_impl(dfdq, wroe, *gamma, dim, stride);
        });
  });
}

template <int N>
void vic_solve_cuda(at::TensorIterator& iter, double dt, int il, int iu) {
  at::cuda::CUDAGuard device_guard(iter.device());

  AT_DISPATCH_FLOATING_TYPES(iter.dtype(), "vic_solve_cuda", [&]() {
    auto nhydro = at::native::ensure_nonempty_size(iter.output(), 0);
    auto stride = at::native::ensure_nonempty_stride(iter.output(), 0);
    auto ny = nhydro - Index::ICY;

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

      forward_sweep_impl(a, b, c, delta, corr, du, dt, ny, stride, il, iu);
      backward_substitution_impl(a, delta, w, du, ny, stride, il, iu);
    });
  });
}

template <int N>
void alloc_eigen_cuda(c10::ScalarType dtype,
                      char *&a, char *&b, char *&c, char *&delta, char *&corr,
                      int ncol, int nlayer) {
  AT_DISPATCH_FLOATING_TYPES(dtype, "alloc_eigen_cuda", [&]() {
    cudaMalloc(
        (void **)&a,
        sizeof(Eigen::Matrix<scalar_t, N, N, Eigen::RowMajor>) * ncol * nlayer);
    int err = checkCudaError("alloc_eigen_cuda::a");
    TORCH_CHECK(err == 0, "eigen memory allocation error");

    cudaMalloc(
        (void **)&b,
        sizeof(Eigen::Matrix<scalar_t, N, N, Eigen::RowMajor>) * ncol * nlayer);
    err = checkCudaError("alloc_eigen_cuda::b");
    TORCH_CHECK(err == 0, "eigen memory allocation error");

    cudaMalloc(
        (void **)&c,
        sizeof(Eigen::Matrix<scalar_t, N, N, Eigen::RowMajor>) * ncol * nlayer);
    err = checkCudaError("alloc_eigen_cuda::c");
    TORCH_CHECK(err == 0, "eigen memory allocation error");

    cudaMalloc((void **)&delta,
               sizeof(Eigen::Vector<scalar_t, N>) * ncol * nlayer);
    err = checkCudaError("alloc_eigen_cuda::delta");
    TORCH_CHECK(err == 0, "eigen memory allocation error");

    cudaMalloc((void **)&corr,
               sizeof(Eigen::Vector<scalar_t, N>) * ncol * nlayer);
    err = checkCudaError("alloc_eigen_cuda::corr");
    TORCH_CHECK(err == 0, "eigen memory allocation error");
  });
}

void free_eigen_cuda(char *&a, char *&b, char *&c, char *&delta, char *&corr) {
  cudaDeviceSynchronize();
  cudaFree(a);
  cudaFree(b);
  cudaFree(c);
  cudaFree(delta);
  cudaFree(corr);
}

}  // namespace snap

namespace at::native {

REGISTER_CUDA_DISPATCH(call_roe_average, &snap::call_roe_average_cuda);
REGISTER_CUDA_DISPATCH(call_eigen_system, &snap::call_eigen_system_cuda);
REGISTER_CUDA_DISPATCH(call_flux_jacobian, &snap::call_flux_jacobian_cuda);

REGISTER_CUDA_DISPATCH(vic_solve3, &snap::vic_solve_cuda<3>);
REGISTER_CUDA_DISPATCH(vic_solve5, &snap::vic_solve_cuda<5>);

REGISTER_CUDA_DISPATCH(alloc_eigen3, &snap::alloc_eigen_cuda<3>);
REGISTER_CUDA_DISPATCH(alloc_eigen5, &snap::alloc_eigen_cuda<5>);
REGISTER_CUDA_DISPATCH(free_eigen, &snap::free_eigen_cuda);

}  // namespace at::native
