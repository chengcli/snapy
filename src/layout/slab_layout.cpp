// C/C++
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

// fmt
#include <fmt/format.h>

// snap
#include "slab_layout.hpp"

namespace snap {

void SlabLayout::report(std::ostream &os) const {
  os << "px=" << _px << " py=" << _py << " periodic_x=" << _periodic_x
     << " periodic_y=" << _periodic_y << "\n";
  os << " Rank | (rx,ry)\n";
  os << "----------------\n";
  for (int r = 0; r < _px * _py; ++r) {
    os << fmt::format(" {:>3} | ({:>2},{:>2})\n", r, _coords[r].x,
                      _coords[r].y);
  }
}

int SlabLayout::neighbor_rank(int rx, int ry, int dx, int dy) const {
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

}  // namespace snap
