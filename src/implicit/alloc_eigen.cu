// cuda
#include <cuda_runtime.h>

// eigen
#include <Eigen/Dense>

// torch
#include <ATen/Dispatch.h>

// fvm
#include <fvm/util/cuda_utils.h>

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

void free_eigen_cuda(char *&a, char *&b, char *&c, char *&delta, char *&corr) {
  cudaDeviceSynchronize();
  cudaFree(a);
  cudaFree(b);
  cudaFree(c);
  cudaFree(delta);
  cudaFree(corr);
}

template void alloc_eigen_cuda<3>(char *&, char *&, char *&, char *&, char *&,
                                  int, int, c10::ScalarType);
template void alloc_eigen_cuda<5>(char *&, char *&, char *&, char *&, char *&,
                                  int, int, c10::ScalarType);

}  // namespace snap
