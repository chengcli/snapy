#pragma once

// C/C++
#include <cstdio>
#include <cstdlib>

// torch
#include <torch/torch.h>

namespace snap {

//! Study-only cell probe (#260). SNAP_PROBE_CELL="n k j i" selects conserved
//! row n at INTERIOR indices (k, j, i); nothing is printed when unset.
struct ProbeCell {
  bool on = false;
  int n = 0, k = 0, j = 0, i = 0;

  static ProbeCell const& get() {
    static ProbeCell p = [] {
      ProbeCell q;
      if (char const* s = std::getenv("SNAP_PROBE_CELL")) {
        q.on = std::sscanf(s, "%d %d %d %d", &q.n, &q.k, &q.j, &q.i) == 4;
      }
      return q;
    }();
    return p;
  }

  static int& call() {
    static int c = 0;
    return c;
  }
};

//! value of x[row, k, j, i] (absolute indices) as double
inline double probe_at(torch::Tensor const& x, int row, int k, int j, int i) {
  return x.index({row, k, j, i}).item<double>();
}

}  // namespace snap
