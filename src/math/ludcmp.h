#pragma once

// Eigen
#include <Eigen/Dense>

// base
#include <configure.h>

/*! Given a matrix a[0..n-1][0..n-1], this routine replaces it by the LU
 * decomposition of a rowwise permutation of itself. a and n are input. a is
 * output, arranged as in NRIC equation (2.3.14) ; indx[0..n-1] is an output
 * vector that records the row permutation effected by the partial pivoting; d
 * is output as +/- 1 depending on whether the number of row interchanges was
 * evenor odd, respectively. This routine is used in combination with lubksbto
 * solve linear equationsor invert a matrix. adapted from Numerical Recipes in
 * C, 2nd Ed., p. 46.
 */
template <typename T, int N>
inline DISPATCH_MACRO int ludcmp(Eigen::Matrix<T, N, N, Eigen::RowMajor> &a,
                                 int *indx) {
  int i, imax, j, k, d;
  T big, dum, sum, temp;
  T vv[N];

  d = 1;
  for (i = 0; i < N; i++) {
    big = 0.0;
    for (j = 0; j < N; j++)
      if ((temp = fabs(a(i, j))) > big) big = temp;
    if (big == 0.0) {
      printf("Singular matrix in routine ludcmp");
      return 1;
    }
    vv[i] = 1.0 / big;
  }
  for (j = 0; j < N; j++) {
    for (i = 0; i < j; i++) {
      sum = a(i, j);
      for (k = 0; k < i; k++) sum -= a(i, k) * a(k, j);
      a(i, j) = sum;
    }
    big = 0.0;
    for (i = j; i < N; i++) {
      sum = a(i, j);
      for (k = 0; k < j; k++) sum -= a(i, k) * a(k, j);
      a(i, j) = sum;
      if ((dum = vv[i] * fabs(sum)) >= big) {
        big = dum;
        imax = i;
      }
    }
    if (j != imax) {
      for (k = 0; k < N; k++) {
        dum = a(imax, k);
        a(imax, k) = a(j, k);
        a(j, k) = dum;
      }
      d = -d;
      vv[imax] = vv[j];
    }
    indx[j] = imax;
    if (j != N - 1) {
      dum = (1.0 / a(j, j));
      for (i = j + 1; i < N; i++) a(i, j) *= dum;
    }
  }

  return d;
}
