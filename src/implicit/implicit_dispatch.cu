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
#include "tridiag_thomas_impl.h"
#include "implicit_dispatch.hpp"

namespace snap {

template <int N>
void alloc_eigen_cuda(char *&a, char *&b, char *&c, char *&delta, char *&corr,
                      int ncol, int nlayer, c10::ScalarType dtype) {
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

REGISTER_CUDA_DISPATCH(vic_forward3, &snap::vic_forward_cuda<3>);
REGISTER_CUDA_DISPATCH(vic_forward5, &snap::vic_forward_cuda<5>);

REGISTER_CUDA_DISPATCH(alloc_eigen3, &snap::alloc_eigen_cuda<3>);
REGISTER_CUDA_DISPATCH(alloc_eigen5, &snap::alloc_eigen_cuda<5>);

REGISTER_CUDA_DISPATCH(free_eigen, &snap::free_eigen_cuda);

}  // namespace at::native
