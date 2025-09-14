/*
 * Notes on connectivity:
 * - We model only the surface (six 2D faces).
 * - Each face holds a px-by-py processor grid (px==py required by the cubed
 * grid; however code allows px,py).
 * - Global rank = face_major * (py*px) + zorder_rank_within_face.
 *
 * Orientation model across an edge:
 *   From (face f, side s ∈ {L,R,B,T}) we land on (nface, nside),
 *   and the along-edge index is either preserved or reversed.
 *   No "transpose" is required if nside is defined correctly:
 *     - neighbor side L/R varies along neighbor Y (rows)
 *     - neighbor side B/T varies along neighbor X (cols)
 *   This matches p4est's face orientation idea at coarse level.
 *
 * If the geometry requires a different convention (e.g., local axes on faces),
 * just edit the table `CS_FACE_EDGES[6][4]`.
 *
 * WRONG
 * Demo cubed-sphere Z-order connectivity px=2 face=4 (rx,ry)=(0,1)
 * self=18 L=14 R=19 D=16 U=11 UL=18
 * Demo cubed-sphere Z-order connectivity px=2 face=5 (rx,ry)=(0,1)
 * self=22 L=13 R=23 D=20 U=0 UL=15
 */

// C/C++
#include <cstdint>
#include <cstdio>
#include <cstdlib>

// fmt
#include <fmt/format.h>

// snap
#include "cubed_sphere_layout.hpp"

namespace snap {

/* --------------------------
 * Cubed-sphere connectivity
 * --------------------------
 * Face numbering (editable):
 *
 *           4
 *       3   0   1   2
 *           5
 *
 * Sides: 0=L, 1=R, 2=B, 3=T  (left, right, bottom, top)
 * Each entry says: leaving face F via side S,
 * you arrive at (nface, nside) and the along-edge index is reversed? (0/1)
 *
 * IMPORTANT: Different codes choose different local face axes.
 * If your tests show flipped corner order, toggle `rev` for that edge.
 */
enum { SIDE_L = 0, SIDE_R = 1, SIDE_B = 2, SIDE_T = 3 };

struct CSEdge {
  int nface; /* neighbor face id [0..5] */
  int nside; /* neighbor side id (LEFT/RIGHT/BOTTOM/TOP) */
  int rev;   /* 0: preserve along-edge index, 1: reverse */
};

static const CSEdge CS_FACE_EDGES[6][4] = {
    /* face 0: neighbors 3(L),1(R),5(B),4(T) */
    [0] = {/* L */ {3, SIDE_R, 0},
           /* R */ {1, SIDE_L, 0},
           /* B */ {5, SIDE_T, 0},
           /* T */ {4, SIDE_B, 0}},
    /* face 1: neighbors 0(L),2(R),5(B),4(T) */
    [1] = {/* L */ {0, SIDE_R, 0},
           /* R */ {2, SIDE_L, 0},
           /* B */ {5, SIDE_R, 1},
           /* T */ {4, SIDE_R, 0}},
    /* face 2: neighbors 1(L),3(R),5(B),4(T) */
    [2] = {/* L */ {1, SIDE_R, 0},
           /* R */ {3, SIDE_L, 0},
           /* B */ {5, SIDE_B, 1},
           /* T */ {4, SIDE_T, 1}},
    /* face 3: neighbors 2(L),0(R),5(B),4(T) */
    [3] = {/* L */ {2, SIDE_R, 0},
           /* R */ {0, SIDE_L, 0},
           /* B */ {5, SIDE_L, 0},
           /* T */ {4, SIDE_L, 1}},
    /* face 4: neighbors 3(L),1(R),0(B),2(T) */
    [4] = {/* L */ {3, SIDE_T, 1},
           /* R */ {1, SIDE_T, 0},
           /* B */ {0, SIDE_T, 0},
           /* T */ {2, SIDE_T, 1}},
    /* face 5: neighbors 3(L),1(R),2(B),0(T) */
    [5] = {/* L */ {3, SIDE_B, 0},
           /* R */ {1, SIDE_B, 1},
           /* B */ {2, SIDE_B, 1},
           /* T */ {0, SIDE_B, 0}}};

static inline void cs_clamp_inside(int pxy, int *nx, int *ny) {
  if (*nx < 0)
    *nx = 0;
  else if (*nx >= pxy)
    *nx = pxy - 1;
  if (*ny < 0)
    *ny = 0;
  else if (*ny >= pxy)
    *ny = pxy - 1;
}

static inline void cs_edge_map_into_neighbor(int pxy, int leaving_side,
                                             int pos /*0..k-1*/,
                                             const CSEdge *emap, int *out_rx,
                                             int *out_ry) {
  /* Map along-edge index into neighbor face border, with optional reversal. */
  int pos2 = pos;
  if (emap->rev) {
    pos2 = pxy - 1 - pos;
  }

  switch (emap->nside) {
    case SIDE_L:
      *out_rx = 0;
      *out_ry = pos2;
      break; /* varies in y */
    case SIDE_R:
      *out_rx = pxy - 1;
      *out_ry = pos2;
      break;
    case SIDE_B:
      *out_rx = pos2;
      *out_ry = 0;
      break; /* varies in x */
    case SIDE_T:
      *out_rx = pos2;
      *out_ry = pxy - 1;
      break;
    default:
      *out_rx = 0;
      *out_ry = 0;
      break;
  }
}

void CubedSphereLayout::report(std::ostream &os) const {
  os << "pxy=" << _pxy << "\n";
  for (int f = 0; f < 6; ++f) {
    os << " Face " << f << "\n";
    os << " Rank | (rx,ry)\n";
    os << "----------------\n";
    for (int r = 0; r < _pxy * _pxy; ++r) {
      os << fmt::format(" {:>3} | ({:>2},{:>2})\n", _rankof6[f][r],
                        _coords6[f][r].x, _coords6[f][r].y);
    }
  }
}

void CubedSphereLayout::step_one(int face, int rx, int ry, int dx, int dy,
                                 int *out_face, int *out_rx,
                                 int *out_ry) const {
  /* Try to stay on-face */
  int nx = rx + dx;
  int ny = ry + dy;
  if (0 <= nx && nx < _pxy && 0 <= ny && ny < _pxy) {
    *out_face = face;
    *out_rx = nx;
    *out_ry = ny;
    return;
  }

  /* Identify which single edge is crossed */
  int side = -1;
  if (nx < 0)
    side = SIDE_L;
  else if (nx >= _pxy)
    side = SIDE_R;
  else if (ny < 0)
    side = SIDE_B;
  else if (ny >= _pxy)
    side = SIDE_T;

  const CSEdge emap = CS_FACE_EDGES[face][side];
  *out_face = emap.nface;

  /* Along-edge position on current face */
  int pos = (side == SIDE_L || side == SIDE_R) ? ry : rx;

  /* Map to neighbor border */
  cs_edge_map_into_neighbor(_pxy, side, pos, &emap, out_rx, out_ry);

  // cs_clamp_inside(pxy, out_rx, out_ry);
}

/* get neighbor GLOBAL rank for (dx,dy) in {-1,0,1}^2 (incl. corners) */
int CubedSphereLayout::neighbor_rank(int face, int rx, int ry, int dx,
                                     int dy) const {
  if (dx == 0 && dy == 0) {
    /* self */
    int rloc = face_local_rank(face, rx, ry);
    return global_rank_from_face_local(face, rloc);
  }

  /* 1-step edge move */
  if ((dx == 0) ^ (dy == 0)) {
    int f1, x1, y1;
    step_one(face, rx, ry, dx, dy, &f1, &x1, &y1);
    int rloc = face_local_rank(f1, x1, y1);
    return global_rank_from_face_local(f1, rloc);
  }

  /* corners: at least crossing one edge, maybe two */
  // find the current block's logical location
  int lx, ly;
  logical_loc2(rx, ry, _pxy, _pxy, &lx, &ly);

  if ((dx + lx <= 1) && (dx + lx >= -1)) {
    // do (dx,0) and then (0,dy)
    // printf("lx = %d, ly = %d, dx = %d, dy = %d\n", lx, ly, dx, dy);
    int f1, x1, y1;
    step_one(face, rx, ry, dx, 0, &f1, &x1, &y1);

    int f2, x2, y2;
    step_one(f1, x1, y1, 0, dy, &f2, &x2, &y2);
    int rloc = face_local_rank(f2, x2, y2);
    return global_rank_from_face_local(f2, rloc);
  } else if ((dy + ly <= 1) && (dy + ly >= -1)) {
    // do (0, dy) and then (dx, 0)
    int f1, x1, y1;
    step_one(face, rx, ry, 0, dy, &f1, &x1, &y1);

    int f2, x2, y2;
    step_one(f1, x1, y1, dx, 0, &f2, &x2, &y2);
    int rloc = face_local_rank(f2, x2, y2);
    return global_rank_from_face_local(f2, rloc);
  } else {  // crossing two edges
    int f1, x1, y1;
    step_one(face, rx, ry, dx, 0, &f1, &x1, &y1);
    int rloc = face_local_rank(f1, x1, y1);
    return global_rank_from_face_local(f1, rloc);
  }
}

}  // namespace snap
