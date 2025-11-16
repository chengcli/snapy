// fmt
#include <fmt/format.h>

// snap
#include "layout.hpp"

namespace snap {

void SlabLayout::report(std::ostream &os) const {
  options.report(os);
  os << " Rank | (rx,ry)\n";
  os << "----------------\n";
  for (int r = 0; r < _px * _py; ++r) {
    os << fmt::format(" {:>3} | ({:>2},{:>2})\n", r, _coords2[r].x,
                      _coords2[r].y);
  }
}

std::tuple<int, int, int> SlabLayout::loc_of(int rank) const {
  if (rank < 0 || rank >= opitions.px() * options.py()) return {-1, -1, 0};
  return {_coords2[rank].x, _coords2[rank].y, 0};
}

int SlabLayout::neighbor_rank(int rx, int ry, int rz, int dx, int dy,
                              int dz) const {
  if (rz != 0 || dz != 0) {
    throw std::runtime_error(
        "SlabLayout::neighbor_rank: rz and dz must be zero in slab layout");
  }

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
