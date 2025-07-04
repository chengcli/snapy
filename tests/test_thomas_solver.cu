// external
#include <gtest/gtest.h>

// snap
#include <snap/implicit/implicit_dispatch.hpp>
#include <snap/implicit/tridiag_thomas_impl.h>

using namespace snap;

#define A(j, i) \
  reinterpret_cast<Eigen::Matrix<T, N, N> *>(a)[(j) * nlayer + (i)]

template <typename T, int N>
__global__ void set_eigen_values(char *a, int ncol, int nlayer) {
  int jcol = blockIdx.x * blockDim.x + threadIdx.x;
  if (jcol >= ncol) return;

  for (int i = 0; i < nlayer; ++i) {
    A(jcol, i) = i * Eigen::Matrix<T, N, N>::Identity();
  }
}

TEST(TridiagThomas, forward_sweep) {
  char *a, *b, *c, *delta, *corr;
  double *du, *w;

  TridiagSolverOptions op;

  op.nhydro(5);
  op.ncol(10000);
  op.nlayer(40);
  op.il(0);
  op.iu(9);

  float dt = 0.1;

  int nhydro = op.nhydro();
  int ncol = op.ncol();
  int nlayer = op.nlayer();

  at::native::alloc_eigen5(c10::kCUDA, c10::kDouble, a, b, c, delta, corr, ncol, nlayer);

  cudaMalloc((void **)&du, sizeof(double) * nhydro * ncol * nlayer);
  cudaMalloc((void **)&w, sizeof(double) * nhydro * ncol * nlayer);

  std::cout << sizeof(Eigen::Matrix<double, 5, 5>) << std::endl;

  set_eigen_values<double, 5><<<(ncol + 511) / 512, 512>>>(a, ncol, nlayer);

  Eigen::Matrix<double, 5, 5> *aa;
  aa = new Eigen::Matrix<double, 5, 5>[ncol * nlayer];
  cudaMemcpy(aa, a, sizeof(Eigen::Matrix<double, 5, 5>) * ncol * nlayer,
             cudaMemcpyDeviceToHost);
  std::cout << aa[41] << std::endl;

  forward_sweep_cuda<double, 5>
      <<<(ncol + 511) / 512, 512>>>(a, b, c, delta, corr, du, dt, op);

  backward_substitution_cuda<double, 5>
      <<<(ncol + 511) / 512, 512>>>(a, delta, w, du, op);

  at::native::free_eigen(c10::kCUDA, a, b, c, delta, corr);

  cudaFree(du);
  cudaFree(w);
  delete[] aa;
}

int main(int argc, char **argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
