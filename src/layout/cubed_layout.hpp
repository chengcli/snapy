#pragma once

// C/C++
#include <sstream>
#include <tuple>

// snap
#include "connectivity.hpp"

namespace snap {

class CubedLayout {
 public:
  CubedLayout(int px, int py, int pz, bool periodic_x = false,
              bool periodic_y = false, bool periodic_z = false)
      : _px(px),
        _py(py),
        _pz(pz),
        _periodic_x(periodic_x),
        _periodic_y(periodic_y),
        _periodic_z(periodic_z) {
    int P = _px * _py * _pz;
    _coords = new Coord3[P];
    _rankof = new int[P];
    build_zorder_coords3(px, py, pz, _coords);
    build_rank_of3(px, py, pz, _coords, _rankof);
  }

  ~CubedLayout() {
    delete[] _coords;
    delete[] _rankof;
  }

  void report(std::ostream &os) const;

  std::tuple<int, int, int> get_procs() const { return {_px, _py, _pz}; }

  int rank_of(int rx, int ry, int rz) const {
    if (rx < 0 || rx >= _px || ry < 0 || ry >= _py || rz < 0 || rz >= _pz)
      return -1;
    return _rankof[rz * (_px * _py) + ry * _px + rx];
  }

  std::tuple<int, int, int> loc_of(int rank) const {
    if (rank < 0 || rank >= _px * _py * _pz) return {-1, -1, -1};
    return {_coords[rank].x, _coords[rank].y, _coords[rank].z};
  }

  /* ============================
   * Neighbor → Z-order rank (3D)
   * ============================
   * dx,dy,dz ∈ {-1,0,1}. periodic flags control wrap; otherwise off-domain →
   * -1. (rx,ry,rz) are THIS rank's coords in the process grid (not Morton
   * code).
   */
  int neighbor_rank(int rx, int ry, int rz, int dx, int dy, int dz) const;

 private:
  int _px, _py, _pz; /* processors per dimension */
  bool _periodic_x, _periodic_y, _periodic_z;
  Coord3 *_coords;
  int *_rankof;
};

}  // namespace snap
