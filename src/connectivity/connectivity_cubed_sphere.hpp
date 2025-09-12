#pragma once

#include "connectivity.hpp"

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

  int get_pxy() const { return _pxy; }

  /* Global rank layout: face-major, Z-order within face */
  size_t global_rank_from_face_local(int face, int r_local) const {
    int P = _pxy * _pxy;
    return (size_t)face * P + (size_t)r_local;
  }

  /* Reverse: get (face, r_local) from global rank */
  void global_rank_to_face_local(size_t grank, int *face, int *r_local) const {
    int P = _pxy * _pxy;
    *face = (int)(grank / P);
    *r_local = (int)(grank % P);
  }

  /* --------------------------
   * Edge stepping helper
   * --------------------------
   * Move off the face by one tile in (dx,dy) ∈ {-1,0,1}^2.
   * Returns neighbor (nface, nrank) or (-1, -1) on error (should not happen on
   * a closed cube).
   *
   * Logic:
   * - If inside same face: trivial offset of (rx,ry).
   * - If crossing a single edge (|dx|+|dy|==1): use edge table to decide
   * neighbor face & side, compute the along-edge index (pos), reverse if
   * needed, and place at neighbor border.
   * - If crossing a corner (|dx|==1 && |dy|==1): do it in two hops (dx,0) then
   * (0,dy) through the intermediate face. This mirrors typical ghost-corner
   * exchange.
   */
  int face_local_rank(int face, int rx, int ry) const {
    int P = _pxy * _pxy;
    /* map local (rx,ry) to per-face Z-order rank */
    return _rankof6[face][linear_index2(_pxy, _pxy, ry, rx)];
  }

  void step_one(int face, int rx, int ry, int dx, int dy, int *out_face,
                int *out_rx, int *out_ry) const;

  size_t neighbor_global_rank(int face, int rx, int ry, int dx, int dy) const;

 private:
  int _pxy;            /* processors per face */
  Coord2 *_coords6[6]; /* coords per face: length P each */
  int *_rankof6[6];    /* inverse map per face: length P each */
};
