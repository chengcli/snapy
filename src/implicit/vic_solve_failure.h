#pragma once

#include <configure.h>

#include <Eigen/Dense>
#include <limits>

namespace snap {

// A failed column is never back-substituted or redistributed. This sentinel
// lives in the existing solution buffer, avoiding separate CPU/CUDA status
// channels; the host module consumes it before any update is applied.
template <typename T, int N>
bool DISPATCH_MACRO vic_fail_column(Eigen::Matrix<T, N, 1>* delta, int il,
                                    int iu) {
  for (int i = il; i <= iu; ++i)
    delta[i].setConstant(std::numeric_limits<T>::quiet_NaN());
  return false;
}

}  // namespace snap
