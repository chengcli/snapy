// eigen
#include <Eigen/Dense>

// torch
#include <ATen/Dispatch.h>

namespace snap {

template void alloc_eigen_cpu<3>(char *&, char *&, char *&, char *&, char *&,
                                 int, int, c10::ScalarType);
template void alloc_eigen_cpu<5>(char *&, char *&, char *&, char *&, char *&,
                                 int, int, c10::ScalarType);
}  // namespace snap
