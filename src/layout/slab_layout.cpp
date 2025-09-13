// C/C++
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

// canoe
#include "slab_layout.hpp"

namespace canoe {

int SlabLayout::neighbor_rank(int rx, int ry, int dx, int dy) {
  int nx = rx + dx;
  int ny = ry + dy;

  if (_periodic_x) {
    if (nx < 0)
      nx += _px;
    else if (nx >= _px)
      nx -= _px;
  } else {
    if (nx < 0 || nx >= _px) return -1;
  }

  if (_periodic_y) {
    if (ny < 0)
      ny += _py;
    else if (ny >= _py)
      ny -= _py;
  } else {
    if (ny < 0 || ny >= _py) return -1;
  }

  return _rankof[linear_index2(_px, _py, ny, nx)];
}

}  // namespace canoe
