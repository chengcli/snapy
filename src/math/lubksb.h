#pragma once

// base
#include <configure.h>

// snap
#include "ludcmp.h"

//! \brief Solve linear equations using LU decomposition backsubstitution
//!
//! Solves the set of n linear equations A X = B using LU decomposition.
//! This routine takes into account the possibility that b will begin with many
//! zero elements, so it is efficient for use in matrix inversion.
//!
//! \tparam T Scalar type (e.g., float, double)
//! \tparam N Matrix dimension
//! \param[in] a LU decomposition of matrix A (from ludcmp), not modified
//! \param[in] indx Permutation vector from ludcmp, not modified
//! \param[in,out] b Right-hand side vector B on input, solution vector X on
//! output
//!
//! \note Adapted from Numerical Recipes in C, 2nd Ed., p. 47.
//! \note a, n, and indx are not modified and can be reused for successive calls
template <typename T, int N>
inline DISPATCH_MACRO void lubksb(
    Eigen::Matrix<T, N, N, Eigen::RowMajor> const &a, int const *indx,
    Eigen::Vector<T, N> &b) {
  int i, ii = 0, ip, j;
  T sum;

  for (i = 0; i < N; i++) {
    ip = indx[i];
    sum = b[ip];
    b[ip] = b[i];
    if (ii)
      for (j = ii - 1; j < i; j++) sum -= a(i, j) * b(j);
    else if (sum)
      ii = i + 1;
    b[i] = sum;
  }
  for (i = N - 1; i >= 0; i--) {
    sum = b[i];
    for (j = i + 1; j < N; j++) sum -= a(i, j) * b(j);
    b[i] = sum / a(i, i);
  }
}
