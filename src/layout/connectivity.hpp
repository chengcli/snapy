#pragma once

// C/C++
#include <cstdint>
#include <cstdlib>

namespace snap {

/* ===========================
 * Bit de-interleaving helpers
 * ===========================
 * compact1by1:  remove every other bit (keep bits at positions 0,2,4,...)
 * compact1by2:  remove two other bits (keep bits at positions 0,3,6,...) for
 * 3-way interleave
 *
 * Based on "Bit Twiddling Hacks" (public domain).
 */

inline int compact1by1(int v) {
  v &= 0x55555555u;
  v = (v | (v >> 1)) & 0x33333333u;
  v = (v | (v >> 2)) & 0x0F0F0F0Fu;
  v = (v | (v >> 4)) & 0x00FF00FFu;
  v = (v | (v >> 8)) & 0x0000FFFFu;
  return v;
}

/* For 3D Morton codes we need to extract every 3rd bit. Using 64-bit lanes. */
inline int compact1by2(uint64_t v) {
  v &= 0x1249249249249249ULL;  // 0b001001.. pattern
  v = (v ^ (v >> 2)) & 0x10c30c30c30c30c3ULL;
  v = (v ^ (v >> 4)) & 0x100f00f00f00f00fULL;
  v = (v ^ (v >> 8)) & 0x1f0000ff0000ffULL;
  v = (v ^ (v >> 16)) & 0x1f00000000ffffULL;
  v = (v ^ (v >> 32)) & 0x1fffffULL;
  return (int)v;
}

/* ===========================
 * Morton decode (Z-order)
 * =========================== */

/* Decode 2D Morton code -> (y,x). Note the order: (y,x). */
inline void morton_decode2(int code, int *y, int *x) {
  *x = compact1by1(code);
  *y = compact1by1(code >> 1);
}

/* Decode 3D Morton code -> (z,y,x). Uses 64-bit Morton codes. */
inline void morton_decode3(uint64_t code, int *z, int *y, int *x) {
  *x = compact1by2(code);
  *y = compact1by2(code >> 1);
  *z = compact1by2(code >> 2);
}

/* ===========================
 * Coordinate containers
 * =========================== */

struct Coord2 {
  int y, x;
};
struct Coord3 {
  int z, y, x;
};

/* ================
 * Z-order builders
 * ================ */

/* Build Py×Px coordinates in Z-order. coords must have length >= py*px. */
size_t build_zorder_coords2(int px, int py, Coord2 *coords);

/* Build Pz×Py×Px coordinates in Z-order. coords must have length >= pz*py*px.
 */
size_t build_zorder_coords3(int px, int py, int pz, Coord3 *coords);

/* ======================
 * coords -> rank mapping
 * ======================
 * We build a dense array rank_of with shape:
 *  - 2D: [py][px]
 *  - 3D: [pz][py][px]
 * storing the Z-order rank at that coordinate.
 * Access via linear index helpers below.
 */

inline size_t linear_index2(int px, int /*py*/, int y, int x) {
  return (size_t)y * (size_t)px + (size_t)x;
}

inline size_t linear_index3(int px, int py, int z, int y, int x) {
  return ((size_t)z * (size_t)py + (size_t)y) * (size_t)px + (size_t)x;
}

/* rank_of2: array length py*px, filled with rank at (y,x) */
void build_rank_of2(int px, int py, const Coord2 *coords, int *rank_of_out);

/* rank_of3: array length pz*py*px, filled with rank at (z,y,x) */
void build_rank_of3(int px, int py, int pz, const Coord3 *coords,
                    int *rank_of_out);

/* Return logical location of a tile on a face in (-1,0,1) coding.
 *   (-1,0) = left edge, (1,0) = right edge, (0,-1) = bottom edge, (0,1) = top
 * edge
 *   (-1,-1) = bottom-left corner, etc.
 *   (0,0) = interior (not on any edge).
 */
static inline void logical_loc2(int rx, int ry, int px, int py, int *lx,
                                int *ly) {
  *lx = 0;
  *ly = 0;

  if (rx == 0)
    *lx = -1;
  else if (rx == px - 1)
    *lx = 1;

  if (ry == 0)
    *ly = -1;
  else if (ry == py - 1)
    *ly = 1;
}

/* Boolean: is this tile on any edge? */
static inline int is_edge_loc(int lx, int ly) { return (lx != 0 || ly != 0); }

/* Boolean: is this tile specifically a corner? */
static inline int is_corner_loc(int lx, int ly) { return (lx != 0 && ly != 0); }

}  // namespace snap
