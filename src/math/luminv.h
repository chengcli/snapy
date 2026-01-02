#pragma once

// base
#include <configure.h>

// snap
#include "lubksb.h"

//! \brief Compute matrix inverse using LU decomposition
//!
//! Using LU decomposition and backsubstitution routines, this function finds
//! the inverse of a matrix column by column.
//!
//! \tparam T Scalar type (e.g., float, double)
//! \tparam N Matrix dimension
//! \param[in] a LU decomposition of the matrix (from ludcmp), not modified
//! \param[in] indx Permutation vector from ludcmp, not modified
//! \param[out] y Output matrix containing the inverse
template <typename T, int N>
inline DISPATCH_MACRO void luminv(
    Eigen::Matrix<T, N, N, Eigen::RowMajor> const &a, int const *indx,
    Eigen::Matrix<T, N, N, Eigen::RowMajor> &y) {
  for (int j = 0; j < N; j++) {
    Eigen::Matrix<T, N, 1> col;
    for (int i = 0; i < N; i++) col(i) = 0.0;
    col(j) = 1.0;
    lubksb(a, indx, col);
    for (int i = 0; i < N; i++) y(i, j) = col(i);
  }
}
