// C/C++
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

// fmt
#include <fmt/format.h>

// snap
#include "cubed_layout.hpp"

namespace snap {

void CubedLayout::report(std::ostream &os) const {
  os << "px=" << _px << " py=" << _py << " pz=" << _pz
     << " periodic_x=" << _periodic_x << " periodic_y=" << _periodic_y
     << " periodic_z=" << _periodic_z << "\n";
  os << " Rank | (rx,ry,rz)\n";
  os << "-------------------\n";
  for (int r = 0; r < _px * _py * _pz; ++r) {
    os << fmt::format(" {:>3} | ({:>2},{:>2},{:>2})\n", r, _coords[r].x,
                      _coords[r].y, _coords[r].z);
  }
}

int CubedLayout::neighbor_rank(int rx, int ry, int rz, int dx, int dy,
                               int dz) const {
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

}  // namespace snap
