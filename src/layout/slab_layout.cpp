// fmt
#include <fmt/format.h>

// snap
#include "layout.hpp"

namespace snap {

void SlabLayoutImpl::report(std::ostream &os) const {
  options.report(os);
  os << " Rank | (rx,ry)\n";
  os << "----------------\n";
  for (int r = 0; r < options.px() * options.py(); ++r) {
    os << fmt::format(" {:>3} | ({:>2},{:>2})\n", r, _coords2[r].x,
                      _coords2[r].y);
  }
}

std::tuple<int, int, int> SlabLayoutImpl::loc_of(int rank) const {
  if (rank < 0 || rank >= options.px() * options.py()) return {-1, -1, 0};
  return {_coords2[rank].x, _coords2[rank].y, 0};
}

int SlabLayoutImpl::neighbor_rank(int rx, int ry, int rz, int dx, int dy,
                                  int dz) const {
  if (rz != 0 || dz != 0) {
    throw std::runtime_error(
        "SlabLayout::neighbor_rank: rz and dz must be zero in slab layout");
  }

  int nx = rx + dx;
  int ny = ry + dy;

  if (options.periodic_x()) {
    if (nx < 0)
      nx += options.px();
    else if (nx >= options.px())
      nx -= options.px();
  } else {
    if (nx < 0 || nx >= options.px()) return -1;
  }

  if (options.periodic_y()) {
    if (ny < 0)
      ny += options.py();
    else if (ny >= options.py())
      ny -= options.py();
  } else {
    if (ny < 0 || ny >= options.py()) return -1;
  }

  return _rankof[linear_index2(options.px(), options.py(), ny, nx)];
}

}  // namespace snap
