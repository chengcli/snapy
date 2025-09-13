// C/C++
#include "cubed_layout.hpp"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace canoe {

int CubedLayout::neighbor_rank(int rx, int ry, int rz, int dx, int dy, int dz) {
  int nx = rx + dx;
  int ny = ry + dy;
  int nz = rz + dz;

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

  if (_periodic_z) {
    if (nz < 0)
      nz += _pz;
    else if (nz >= _pz)
      nz -= _pz;
  } else {
    if (nz < 0 || nz >= _pz) return -1;
  }

  return _rankof[linear_index3(_px, _py, nz, ny, nx)];
}

}  // namespace canoe
