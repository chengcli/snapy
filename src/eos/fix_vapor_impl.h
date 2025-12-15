#pragma once

// base
#include <configure.h>

namespace snap {

template <typename T>
inline DISPATCH_MACRO int fix_vapor_impl(T* vapor, T const* major, int nx1) {
  int is = nx1 - 1;
  int ie = 0;

  // scan from top to bottom
  while (is >= ie) {
    if (major[is] <= 0.) return 1;  // fail
    if (vapor[is] > 0.) {
      is--;
      continue;
    }

    // find next valid
    int i = is;
    T sum_vapor = vapor[i];
    T sum_major = major[i];
    while (sum_vapor <= 0. && i >= ie) {
      sum_vapor += vapor[i];
      sum_major += major[i];
      i--;
    }

    if (i < ie && sum_vapor <= 0.) return 1;  // fail

    // redistribute concentrations from is (inclusive) to i (exclusive)
    T yfrac = sum_vapor / sum_major;
    for (int j = is; j > i; --j) {
      vapor[j] = yfrac * major[j];
    }
    // printf("yfrac = %e redistributed from %d to %d\n", double(yfrac), i + 1,
    // is);

    // continue scan
    is = i;
  }

  return 0;
}

}  // namespace snap
