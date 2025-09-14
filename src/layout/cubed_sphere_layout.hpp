#pragma once

// C/C++
#include <sstream>
#include <tuple>

// snap
#include "connectivity.hpp"

namespace snap {

/* --------------------------
 * Per-face Z-order layout
 * -------------------------- */
class CubedSphereLayout {
 public:
  CubedSphereLayout(int pxy) : _pxy(pxy) {
    int P = _pxy * _pxy;
    for (int f = 0; f < 6; ++f) {
      _coords6[f] = new Coord2[P];
      _rankof6[f] = new int[P];
      build_zorder_coords2(_pxy, _pxy, _coords6[f]);
      build_rank_of2(_pxy, _pxy, _coords6[f], _rankof6[f]);
    }
  }

  ~CubedSphereLayout() {
    for (int f = 0; f < 6; ++f) {
      delete[] _coords6[f];
      delete[] _rankof6[f];
    }
  }

  void report(std::ostream &os) const;

  int get_procs() const { return _pxy; }

  int rank_of(int face, int rx, int ry) const {
    if (face < 0 || face >= 6) return -1;
    if (rx < 0 || rx >= _pxy || ry < 0 || ry >= _pxy) return -1;
    return _rankof6[face][ry * _pxy + rx];
  }

  std::tuple<int, int, int> loc_of(int global_rank) const {
    if (global_rank < 0 || global_rank >= 6 * _pxy * _pxy) return {-1, -1, -1};
    int face, r_local;
    global_rank_to_face_local(global_rank, &face, &r_local);
    int rx = _coords6[face][r_local].x;
    int ry = _coords6[face][r_local].y;
    return {face, rx, ry};
  }

  /* Global rank layout: face-major, Z-order within face */
  int global_rank_from_face_local(int face, int r_local) const {
    int P = _pxy * _pxy;
    return face * P + r_local;
  }

  /* Reverse: get (face, r_local) from global rank */
  void global_rank_to_face_local(int grank, int *face, int *r_local) const {
    int P = _pxy * _pxy;
    *face = grank / P;
    *r_local = grank % P;
  }

  /* ==========================
   * Edge stepping helper
   * ==========================
   * Move off the face by one tile in (dx,dy) ∈ {-1,0,1}^2.
   * Returns neighbor (nface, nrank) or (-1, -1) on error (should not happen on
   * a closed cube).
   *
   * Logic:
   * - If inside same face: trivial offset of (rx,ry).
   * - If crossing a single edge (|dx|+|dy|==1): use edge table to decide
   *    neighbor face & side, compute the along-edge index (pos), reverse if
   *    needed, and place at neighbor border.
   * - If crossing a corner (|dx|==1 && |dy|==1): do it in two hops.
   *    (dx,0) and (0,dy) through the intermediate face.
   *    If across a panel boundary, do first step inside the panel
   *    and second step outside. This mirrors typical ghost-corner
   *    exchange.
   */
  int face_local_rank(int face, int rx, int ry) const {
    /* map local (rx,ry) to per-face Z-order rank */
    return _rankof6[face][linear_index2(_pxy, _pxy, ry, rx)];
  }

  void step_one(int face, int rx, int ry, int dx, int dy, int *out_face,
                int *out_rx, int *out_ry) const;

  /* ============================
   * Neighbor → Z-order rank (2D)
   * ============================
   * dx,dy ∈ {-1,0,1}. periodic flags control wrap; otherwise off-domain → -1.
   * (rx,ry) are THIS rank's coords in the process grid (not Morton code).
   */
  int neighbor_rank(int face, int rx, int ry, int dx, int dy) const;

 private:
  int _pxy;            /* processors per face */
  Coord2 *_coords6[6]; /* coords per face: length P each */
  int *_rankof6[6];    /* inverse map per face: length P each */
};

}  // namespace snap
