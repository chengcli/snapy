// cuda
#include <cuda_runtime.h>

// eigen
#include <Eigen/Dense>

// torch
#include <ATen/Dispatch.h>

// snap
#include <snap/utils/cuda_utils.h>

namespace snap {


template void alloc_eigen_cuda<3>(char *&, char *&, char *&, char *&, char *&,
                                  int, int, c10::ScalarType);
template void alloc_eigen_cuda<5>(char *&, char *&, char *&, char *&, char *&,
                                  int, int, c10::ScalarType);

}  // namespace snap
