#pragma once

// C/C++
#include <sstream>
#include <utility>

// snap
#include "connectivity.hpp"

namespace snap {

class SlabLayout {
 public:
  SlabLayout(int px, int py, bool periodic_x = false, bool periodic_y = false)
      : _px(px), _py(py), _periodic_x(periodic_x), _periodic_y(periodic_y) {
    int P = _px * _py;
    _coords = new Coord2[P];
    _rankof = new int[P];
    build_zorder_coords2(px, py, _coords);
    build_rank_of2(px, py, _coords, _rankof);
  }

  ~SlabLayout() {
    delete[] _coords;
    delete[] _rankof;
  }

  void report(std::ostream &os) const;

  std::pair<int, int> get_procs() const { return {_px, _py}; }

  int rank_of(int rx, int ry) const {
    if (rx < 0 || rx >= _px || ry < 0 || ry >= _py) return -1;
    return _rankof[ry * _px + rx];
  }

  std::pair<int, int> loc_of(int rank) const {
    if (rank < 0 || rank >= _px * _py) return {-1, -1};
    return {_coords[rank].x, _coords[rank].y};
  }

  /* ============================
   * Neighbor → Z-order rank (2D)
   * ============================
   * dx,dy ∈ {-1,0,1}. periodic flags control wrap; otherwise off-domain → -1.
   * (rx,ry) are THIS rank's coords in the process grid (not Morton code).
   */
  int neighbor_rank(int rx, int ry, int dx, int dy) const;

 private:
  int _px, _py; /* processors per dimension */
  bool _periodic_x, _periodic_y;
  Coord2 *_coords;
  int *_rankof;
};

}  // namespace snap
