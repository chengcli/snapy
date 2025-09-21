#pragma once

// C/C++
#include "cubed_sphere_utils.hpp"

namespace snap {

/* Example usage (pseudo-code at one time step):
   target ghost cell (face_t, side_t, depth, j) <- interpolate from source edge
   interior line
*/
template <typename T>
T sample_source_line(const T *src_face_data, int N, int side_s, T u_src) {
  /* Choose which row/col supplies the line of values */
  int i0, i1; /* integer indices bracketing u_src */
  T u = u_src;
  if (u < 0.0) u = 0.0;
  if (u > (T)(N - 1)) u = (T)(N - 1);

  i0 = (int)floor(u);
  i1 = i0 + 1;
  if (i1 >= N) {
    i1 = N - 1;
    i0 = N - 1;
  } /* clamp right end */
  T w1 = u - (T)i0;
  T w0 = 1.0 - w1;

  /* Fetch along the appropriate line from the source face’s *interior* */
  /* Suppose src_face_data is stored row-major [y][x], 0..N-1 each, without
   * ghosts. */
  T v0, v1;
  switch (side_s) {
    case SIDE_L: /* use x=0 interior column, vary y=i */
      v0 = src_face_data[i0 * N + 0];
      v1 = src_face_data[i1 * N + 0];
      break;
    case SIDE_R: /* x=N-1 interior column */
      v0 = src_face_data[i0 * N + (N - 1)];
      v1 = src_face_data[i1 * N + (N - 1)];
      break;
    case SIDE_B: /* y=0 interior row, vary x=i */
      v0 = src_face_data[0 * N + i0];
      v1 = src_face_data[0 * N + i1];
      break;
    case SIDE_T: /* y=N-1 interior row */
      v0 = src_face_data[(N - 1) * N + i0];
      v1 = src_face_data[(N - 1) * N + i1];
      break;
    default:
      v0 = v1 = 0.0;
  }
  return w0 * v0 + w1 * v1; /* linear; swap in cubic if desired */
}

/* … later, building target ghosts: */
template <typename T>
void fill_one_ghost_cell(
    T *tgt_face_ghost_storage, /* wherever you store ghosts */
    int N, int nghost, const CSGhostMap *table, int face_t, int side_t,
    int depth, int j,
    const T *src_faces[6]) /* pointers to each face’s interior field */
{
  size_t idx = cs_gmap_index(face_t, side_t, depth, j, N, nghost);
  const CSGhostMap *m = &table[idx];

  /* src face interior data (N×N, row-major [y][x]) */
  const T *src = src_faces[m->face_s];

  /* interpolate along source edge interior line */
  T val = sample_source_line(src, N, m->side_s, m->u_src);

  /* write val into your target ghost location … (your layout) */
  (void)tgt_face_ghost_storage; /* left to your array layout */
}
