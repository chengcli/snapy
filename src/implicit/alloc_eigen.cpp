// eigen
#include <Eigen/Dense>

// torch
#include <ATen/Dispatch.h>

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

void free_eigen_cpu(char *&a, char *&b, char *&c, char *&delta, char *&corr) {
  delete[] a;
  delete[] b;
  delete[] c;
  delete[] delta;
  delete[] corr;
}

template void alloc_eigen_cpu<3>(char *&, char *&, char *&, char *&, char *&,
                                 int, int, c10::ScalarType);
template void alloc_eigen_cpu<5>(char *&, char *&, char *&, char *&, char *&,
                                 int, int, c10::ScalarType);
}  // namespace snap
