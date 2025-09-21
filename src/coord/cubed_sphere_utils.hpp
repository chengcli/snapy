#pragma once

// C/C++
#include <cmath>

namespace snap {

struct Vec3 {
  double x, y, z;
};

/* One ghost cell mapping (target ghost -> source 1-D location) */
struct CSGhostMap {
  int64_t face_s;  /* source face id */
  int64_t side_s;  /* source side id (SIDE_L/R/B/T) */
  int64_t j_along; /* along-edge index on target (0..N-1) */
  int64_t depth;   /* ghost depth (1..nghost) */

  double u_src; /* fractional index along the *source* edge interior line */
  double xi_s;  /* optional: source face angle xi (debug/validation) */
  double eta_s; /* optional: source face angle eta (debug/validation) */
};

/* Equiangular centers & ghost centers
 * cell centers: [-pi/4, pi/4], Δ = pi/(2N), center i in [0..N-1]
 */
inline double cs_equ_center(int N, int i) {
  double d = M_PI / (2.0 * (double)N);
  return -M_PI / 4.0 + ((double)i + 0.5) * d;
}

/* Convert angle on a face to a fractional center index (for 1-D interp).
 * If the line varies in Y (L/R sides): use eta. If varies in X (B/T): use xi.
 * Returns u in "cell-center units": 0.0 ~ center 0, 1.0 ~ center 1, ... (N-1)
 */
inline double cs_angle_to_center_u(double angle, int N) {
  double d = M_PI / (2.0 * (double)N);
  /* centers: angle = -pi/4 + (i+0.5)*d  => i = (angle + pi/4)/d - 0.5 */
  return (angle + M_PI / 4.0) / d - 0.5;
}

/* Project unit vector to (xi,eta) on a chosen face (no face selection).
 * Inverse of the above patterns: for +X, a=Y/X, b=Z/X, then xi=atan(a), etc.
 * Assumes the vector is visible on that face (denominator sign consistent).
 */
void cs_vec_to_face_coords(int face, Vec3 v, double *xi, double *eta);

/* Ghost cell centers just outside the panel.
 * side: SIDE_L,SIDE_R,SIDE_B,SIDE_T (left/right/bottom/top)
 * N: cells per dim (px==py==N)
 * j_along: along-edge index (0..N-1) (varies in y for L/R, in x for B/T)
 * o_depth: ghost depth (1..nghost)
 * Out: (xi_t, eta_t) on the target face (ghost location angles)
 */
void cs_equ_ghost_center(int side, int N, int j_along, int o_depth,
                         double *xi_t, double *eta_t);

/* From local (xi,eta) to unit vector on S^2 for a given face.
 * Equiangular gnomonic: a = tan(xi), b = tan(eta).
 * For +X face: (X,Y,Z) ∝ (1, a, b); normalize.
 */
Vec3 cs_face_to_vec(int face, double xi, double eta);

/* map target ghost -> source 1-D coordinate
 * Inputs:
 *   face_t:  target face id
 *   side_t:  target side (SIDE_L/R/B/T)
 *   N:       cells per dim (px==py==N)
 *   j_along: target along-edge index in [0..N-1]
 *   depth_o: ghost depth (1..nghost)
 * Outputs:
 *   *face_s, *side_s: source face and side (from CS_FACE_EDGES)
 *   *u_src:  fractional index along the source edge line (for 1-D interp)
 *   *xi_s, *eta_s: source angles of the mapped ghost point (optional debug)
 *
 * Usage: sample your source data along the edge-aligned interior line
 *        (the row/col adjacent to side_s) at position u_src with 1-D
 * interpolation.
 */
void cs_target_ghost_to_source_u(int face_t, int side_t, int N, int j_along,
                                 int depth_o, int *face_s, int *side_s,
                                 double *u_src, double *xi_s, double *eta_s);

/* Build full ghost->source interpolation table for all faces & edges.
 * Inputs:
 *   N       : cells per dimension on each face (px==py==N)
 *   nghost  : number of ghost layers to fill (>=1)
 *   apply_rev_flag : if nonzero, apply CS_FACE_EDGES[...].rev to flip u_src
 * index
 *
 * Output:
 *   gmap : caller-provided array with length 6 * 4 * nghost * N
 *          (use cs_gmap_index(...) to access)
 */
void cs_build_ghost_map_table(int N, int nghost, int apply_rev_flag,
                              CSGhostMap *gmap);

}  // namespace snap
