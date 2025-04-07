#pragma once

// eigen
#include <Eigen/Dense>

// base
#include <configure.h>

// snap
#include <snap/index.h>
#include <snap/math/ludcmp.h>
#include <snap/math/luminv.h>
#include <snap/util/print_matrix.h>

#define DU(n, i) du[(n) * stride + (i)]
#define W(n, i) w[(n) * stride + (i)]

namespace snap {
template <typename T, int N>
DISPATCH_MACRO void forward_sweep_impl(
    Eigen::Matrix<T, N, N, Eigen::RowMajor> *a,
    Eigen::Matrix<T, N, N, Eigen::RowMajor> *b,
    Eigen::Matrix<T, N, N, Eigen::RowMajor> *c, Eigen::Vector<T, N> *delta,
    Eigen::Vector<T, N> *corr, T *du, double dt, int nhydro, int stride, int il,
    int iu) {
  constexpr int IDN = index::IDN;
  constexpr int IVX = index::IVX;
  constexpr int IVY = index::IVY;
  constexpr int IVZ = index::IVZ;
  constexpr int IPR = index::IPR;
  constexpr int ICY = index::ICY;

  Eigen::Vector<T, N> rhs;

  rhs(0) = DU(IDN, il);
  for (int n = ICY; n < nhydro; ++n) {
    rhs(0) += DU(n, il);
  }
  rhs(0) /= dt;
  rhs(1) = DU(IVX, il) / dt;
  rhs(N - 1) = DU(IPR, il) / dt;

  if (N == 3) {  // partial matrix
    rhs -= corr[il];
  } else {  // full matrix
    rhs(2) = DU(IVY, il) / dt;
    rhs(3) = DU(IVZ, il) / dt;
  }

  int indx[N];
  Eigen::Matrix<T, N, N, Eigen::RowMajor> Y;

  if (N > 4) {
    ludcmp(a[il], indx);
    luminv(a[il], indx, Y);
    a[il] = Y;
  } else {  // Eigen small matrix inverse (N <= 4)
    a[il] = a[il].inverse().eval();
  }
  // printf_matrix("A(%s) = ", a[il], il);
  delta[il] = a[il] * rhs;
  a[il] = a[il] * c[il];

  for (int i = il + 1; i <= iu; ++i) {
    rhs(0) = DU(IDN, i);
    for (int n = ICY; n < nhydro; ++n) {
      rhs(0) += DU(n, i);
    }
    rhs(0) /= dt;
    rhs(1) = DU(IVX, i) / dt;
    rhs(N - 1) = DU(IPR, i) / dt;

    if (N == 3) {  // partial matrix
      rhs -= corr[i];
    } else {
      rhs(2) = DU(IVY, i) / dt;
      rhs(3) = DU(IVZ, i) / dt;
    }

    if (N > 4) {
      a[i] -= b[i] * a[i - 1];
      ludcmp(a[i], indx);
      luminv(a[i], indx, Y);
      a[i] = Y;
    } else {  // Eigen small matrix inverse (N <= 4)
      a[i] = (a[i] - b[i] * a[i - 1]).inverse().eval();
    }

    delta[i] = a[i] * (rhs - b[i] * delta[i - 1]);
    a[i] = a[i] * c[i];
  }
}

template <typename T, int N>
DISPATCH_MACRO void backward_substitution_impl(
    Eigen::Matrix<T, N, N, Eigen::RowMajor> *a, Eigen::Vector<T, N> *delta,
    T *w, T *du, int nhydro, int stride, int il, int iu) {
  constexpr int IDN = index::IDN;
  constexpr int IVX = index::IVX;
  constexpr int IVY = index::IVY;
  constexpr int IVZ = index::IVZ;
  constexpr int IPR = index::IPR;
  constexpr int ICY = index::ICY;

  // update solutions, i=iu
  for (int i = iu - 1; i >= il; --i) {
    delta[i] -= a[i] * delta[i + 1];
  }

  for (int i = il; i <= iu; ++i) {
    auto dens = DU(IDN, i);

    for (int n = IVY; n < nhydro; ++n) {
      dens += DU(n, i);
    }
    dens = delta[i](0) - dens;

    DU(IDN, i) = delta[i](0);
    for (int n = ICY; n < nhydro; ++n) {
      DU(n, i) += dens * W(n, i);
    }
    DU(IVX, i) = delta[i](1);
    DU(IPR, i) = delta[i](N - 1);

    if (N == 5) {  // full matrix
      DU(IVY, i) = delta[i](2);
      DU(IVZ, i) = delta[i](3);
    }

    for (int n = ICY; n < nhydro; ++n) {
      DU(IDN, i) -= DU(n, i);
    }
  }
}
}  // namespace snap

#undef DU
#undef W
