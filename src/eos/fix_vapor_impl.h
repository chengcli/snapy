#pragma once

// base
#include <configure.h>

namespace snap {

template <typename T>
inline DISPATCH_MACRO int fix_vapor_impl(T* vapor, T const* major, T const* vol,
                                         int nx1) {
  int is = nx1 - 1;
  int ie = 0;
  // cell volumes relative to the first cell: exactly 1 on a uniform column
  T const vol0 = vol[0];

  // scan from top to bottom
  while (is >= ie) {
    if (major[is] <= 0.) return 1;  // fail
    if (vapor[is] >= 0.) {
      is--;
      continue;
    }

    // find next valid
    int i = is;
    // Accumulators must start at zero: seeding with vapor[is]/major[is]
    // double-counts cell `is` (the first pass of the loop below adds the same i
    // again), making the redistribution a one-way vapour sink. With zero seeds
    // the accumulation covers exactly the cells (i, is] that are rewritten, so
    // mass is conserved exactly. Both sums are volume-weighted, so the column's
    // mass sum(vapor * vol) is conserved, not sum(vapor) (#241).
    T sum_vapor = 0.;
    T sum_major = 0.;
    do {
      T w = vol[i] / vol0;
      sum_vapor += vapor[i] * w;
      sum_major += major[i] * w;
      i--;
    } while (sum_vapor < 0. && i >= ie);

    if (i < ie && sum_vapor < 0.) {
      // below is exhausted: take the shortfall from above, untouched on failure
      T deficit = -sum_vapor;  // volume-weighted, like the sums
      T above = 0.;
      for (int j = is + 1; j < nx1; ++j) {
        if (major[j] <= 0.) return 1;
        above += vapor[j] * (vol[j] / vol0);
      }
      if (above < deficit) return 1;

      for (int j = is; j >= ie; --j) vapor[j] = 0.;
      for (int j = is + 1; j < nx1 && deficit > 0.; ++j) {
        T w = vol[j] / vol0;
        T need = deficit / w;  // the deficit in cell j's concentration
        if (vapor[j] <= need) {
          deficit -= vapor[j] * w;
          vapor[j] = 0.;
        } else {
          vapor[j] -= need;
          deficit = 0.;
        }
      }
      return 0;
    }

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
