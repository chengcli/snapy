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

#include "connectivity.hpp"

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
/* If your corner order tests fail, adjust the `rev` flags above:
   setting rev=1 flips the along-edge index mapping. */

/* --------------------------
 * Per-face Z-order layout
 * -------------------------- */
struct CSZOrder {
  uint32_t px, py;    /* processors per face in x (cols) and y (rows) */
  size_t P;           /* px*py */
  Coord2 *coords6[6]; /* coords per face: length P each */
  int *rankof6[6];    /* inverse map per face: length P each */
};

static inline void cs_zorder_init(CSZOrder *cs, uint32_t px, uint32_t py) {
  cs->px = px;
  cs->py = py;
  cs->P = (size_t)px * (size_t)py;
  for (int f = 0; f < 6; ++f) {
    cs->coords6[f] = (Coord2 *)malloc(cs->P * sizeof(Coord2));
    cs->rankof6[f] = (int *)malloc(cs->P * sizeof(int));
    build_zorder_coords2(px, py, cs->coords6[f]);
    build_rank_of2(px, py, cs->coords6[f], cs->rankof6[f]);
  }
}

static inline void cs_zorder_destroy(CSZOrder *cs) {
  for (int f = 0; f < 6; ++f) {
    free(cs->coords6[f]);
    free(cs->rankof6[f]);
  }
}

/* Global rank layout: face-major, Z-order within face */
static inline size_t cs_global_rank_from_face_local(const CSZOrder *cs,
                                                    int face, int r_local) {
  return (size_t)face * cs->P + (size_t)r_local;
}

/* Reverse: get (face, r_local) from global rank */
static inline void cs_global_rank_to_face_local(const CSZOrder *cs,
                                                size_t grank, int *face,
                                                int *r_local) {
  *face = (int)(grank / cs->P);
  *r_local = (int)(grank % cs->P);
}

/* --------------------------
 * Edge stepping helper
 * --------------------------
 * Move off the face by one tile in (dx,dy) ∈ {-1,0,1}^2.
 * Returns neighbor (nface, nrank) or (-1, -1) on error (should not happen on a
 * closed cube).
 *
 * Logic:
 * - If inside same face: trivial offset of (rx,ry).
 * - If crossing a single edge (|dx|+|dy|==1): use edge table to decide neighbor
 * face & side, compute the along-edge index (pos), reverse if needed, and place
 * at neighbor border.
 * - If crossing a corner (|dx|==1 && |dy|==1): do it in two hops (dx,0) then
 * (0,dy) through the intermediate face. This mirrors typical ghost-corner
 * exchange.
 */
static inline int cs_face_local_rank(const CSZOrder *cs, int face, uint32_t rx,
                                     uint32_t ry) {
  /* map local (rx,ry) to per-face Z-order rank */
  return cs->rankof6[face][linear_index2(cs->px, cs->py, ry, rx)];
}

static inline void cs_clamp_inside(uint32_t px, uint32_t py, int *nx, int *ny) {
  if (*nx < 0)
    *nx = 0;
  else if (*nx >= (int)px)
    *nx = (int)px - 1;
  if (*ny < 0)
    *ny = 0;
  else if (*ny >= (int)py)
    *ny = (int)py - 1;
}

static inline void cs_edge_map_into_neighbor(
    uint32_t px, uint32_t py, int leaving_side, int pos /*0..k-1*/,
    const CSEdge *emap, uint32_t *out_rx, uint32_t *out_ry) {
  /* Map along-edge index into neighbor face border, with optional reversal. */
  uint32_t pos2;
  if (emap->rev) {
    pos2 = (leaving_side == SIDE_L || leaving_side == SIDE_R)
               ? (py - 1 - (uint32_t)pos)
               : (px - 1 - (uint32_t)pos);
  } else {
    pos2 = (uint32_t)pos;
  }

  switch (emap->nside) {
    case SIDE_L:
      *out_rx = 0;
      *out_ry = pos2;
      break; /* varies in y */
    case SIDE_R:
      *out_rx = px - 1;
      *out_ry = pos2;
      break;
    case SIDE_B:
      *out_rx = pos2;
      *out_ry = 0;
      break; /* varies in x */
    case SIDE_T:
      *out_rx = pos2;
      *out_ry = py - 1;
      break;
    default:
      *out_rx = 0;
      *out_ry = 0;
      break;
  }
}

static inline void cs_step_one(const CSZOrder *cs, int face, uint32_t rx,
                               uint32_t ry, int dx, int dy, int *out_face,
                               uint32_t *out_rx, uint32_t *out_ry) {
  const uint32_t px = cs->px, py = cs->py;

  /* Try to stay on-face */
  int nx = (int)rx + dx;
  int ny = (int)ry + dy;
  if (0 <= nx && nx < (int)px && 0 <= ny && ny < (int)py) {
    *out_face = face;
    *out_rx = (uint32_t)nx;
    *out_ry = (uint32_t)ny;
    return;
  }

  /* Identify which single edge is crossed */
  int side = -1;
  if (nx < 0)
    side = SIDE_L;
  else if (nx >= (int)px)
    side = SIDE_R;
  else if (ny < 0)
    side = SIDE_B;
  else if (ny >= (int)py)
    side = SIDE_T;

  const CSEdge emap = CS_FACE_EDGES[face][side];
  *out_face = emap.nface;

  /* Along-edge position on current face */
  int pos = (side == SIDE_L || side == SIDE_R) ? (int)ry : (int)rx;

  /* Map to neighbor border */
  cs_edge_map_into_neighbor(px, py, side, pos, &emap, out_rx, out_ry);

  /* Finally step one cell inward on neighbor, following (dx,dy) sense */
  /* Convert (dx,dy) to neighbor's inward direction based on nside we landed on
   */
  cs_clamp_inside(px, py, (int *)out_rx, (int *)out_ry);

  /*switch (emap.nside) {
    case SIDE_L:
      printf("In SIDE_L, dx = %d, dy = %d\n", dx, dy);
      if (dx < 0) {
      }
      *out_rx += 0;
      if (dx > 0) {
      };
      if (dy != 0) {
        int t = (int)(*out_ry) + dy;
        out_ry = (uint32_t)t;
        printf("After dy step, out_ry=%u\n", *out_ry);
        cs_clamp_inside(px, py, (int *)out_rx, (int *)out_ry);
      }
      break;
    case SIDE_R:
      if (dx > 0) {
      }
      *out_rx -= 0;
      if (dy != 0) {
        int t = (int)(*out_ry) + dy;
        *out_ry = (uint32_t)t;
        cs_clamp_inside(px, py, (int *)out_rx, (int *)out_ry);
      }
      break;
    case SIDE_B:
      if (dy < 0) {
      }
      *out_ry += 0;
      if (dx != 0) {
        int t = (int)(*out_rx) + dx;
        *out_rx = (uint32_t)t;
        cs_clamp_inside(px, py, (int *)out_rx, (int *)out_ry);
      }
      break;
    case SIDE_T:
      if (dy > 0) {
      }
      *out_ry -= 0;
      if (dx != 0) {
        int t = (int)(*out_rx) + dx;
        *out_rx = (uint32_t)t;
        cs_clamp_inside(px, py, (int *)out_rx, (int *)out_ry);
      }
      break;
  }*/
}

/* Public: get neighbor GLOBAL rank for (dx,dy) in {-1,0,1}^2 (incl. corners) */
static inline long cs_neighbor_global_rank(const CSZOrder *cs, int face,
                                           uint32_t rx, uint32_t ry, int dx,
                                           int dy) {
  if (dx == 0 && dy == 0) {
    /* self */
    int rloc = cs_face_local_rank(cs, face, rx, ry);
    return (long)cs_global_rank_from_face_local(cs, face, rloc);
  }

  /* 1-step edge move */
  if ((dx == 0) ^ (dy == 0)) {
    int f1;
    uint32_t x1, y1;
    cs_step_one(cs, face, rx, ry, dx, dy, &f1, &x1, &y1);
    int rloc = cs_face_local_rank(cs, f1, x1, y1);
    return (long)cs_global_rank_from_face_local(cs, f1, rloc);
  }

  /* corners: at least crossing one edge, maybe two */
  // find the current block's logical location
  int lx, ly;
  cs_logical_loc2(rx, ry, cs->px, cs->py, &lx, &ly);

  if ((dx + lx <= 1) && (dx + lx >= -1)) {
    // do (dx,0) and then (0,dy)
    // printf("lx = %d, ly = %d, dx = %d, dy = %d\n", lx, ly, dx, dy);
    int f1;
    uint32_t x1, y1;
    cs_step_one(cs, face, rx, ry, dx, 0, &f1, &x1, &y1);

    int f2;
    uint32_t x2, y2;
    cs_step_one(cs, f1, x1, y1, 0, dy, &f2, &x2, &y2);
    int rloc = cs_face_local_rank(cs, f2, x2, y2);
    return (long)cs_global_rank_from_face_local(cs, f2, rloc);
  } else if ((dy + ly <= 1) && (dy + ly >= -1)) {
    // do (0, dy) and then (dx, 0)
    int f1;
    uint32_t x1, y1;
    cs_step_one(cs, face, rx, ry, 0, dy, &f1, &x1, &y1);

    int f2;
    uint32_t x2, y2;
    cs_step_one(cs, f1, x1, y1, dx, 0, &f2, &x2, &y2);
    int rloc = cs_face_local_rank(cs, f2, x2, y2);
    return (long)cs_global_rank_from_face_local(cs, f2, rloc);
  } else {  // crossing two edges
    int f1;
    uint32_t x1, y1;
    cs_step_one(cs, face, rx, ry, dx, 0, &f1, &x1, &y1);
    int rloc = cs_face_local_rank(cs, f1, x1, y1);
    return (long)cs_global_rank_from_face_local(cs, f1, rloc);
  }
}

void run_demo(int face, uint32_t px, uint32_t rx, uint32_t ry) {
  printf(
      "Demo cubed-sphere Z-order connectivity px=%u face=%d (rx,ry)=(%u,%u)\n",
      px, face, rx, ry);

  CSZOrder cs;
  cs_zorder_init(&cs, /*px=*/px, /*py=*/px);

  long g_self = cs_neighbor_global_rank(&cs, face, rx, ry, 0, 0);
  long g_left = cs_neighbor_global_rank(&cs, face, rx, ry, -1, 0);
  long g_right = cs_neighbor_global_rank(&cs, face, rx, ry, 1, 0);
  long g_down = cs_neighbor_global_rank(&cs, face, rx, ry, 0, -1);
  long g_up = cs_neighbor_global_rank(&cs, face, rx, ry, 0, 1);
  long g_ul = cs_neighbor_global_rank(&cs, face, rx, ry, -1, 1); /* corner */
  long g_dr = cs_neighbor_global_rank(&cs, face, rx, ry, 1, -1); /* corner */

  printf("self=%ld L=%ld R=%ld D=%ld U=%ld UL=%ld DR=%ld\n", g_self, g_left,
         g_right, g_down, g_up, g_ul, g_dr);

  cs_zorder_destroy(&cs);
}

int main(void) {
  for (int n = 0; n < 6; ++n) {
    printf("\nface %d tests:\n", n);
    run_demo(n, 2, 0, 0);
    run_demo(n, 2, 0, 1);
    run_demo(n, 2, 1, 0);
    run_demo(n, 2, 1, 1);
  }
  return 0;
}
