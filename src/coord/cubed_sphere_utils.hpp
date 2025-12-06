#pragma once

// C/C++
#include <cmath>
#include <cstdint>
#include <vector>

// snap
#include <snap/layout/cubed_sphere_layout.hpp>

namespace snap {

struct Vec3 {
  double x, y, z;
};

/*!
 * Equiangular centers & ghost centers
 * cell centers: [-pi/4, pi/4], d = pi/(2N), center i in [0..N-1]
 */
inline double cs_equ_center(int N, int i) {
  double d = M_PI / (2.0 * (double)N);
  return -M_PI / 4.0 + ((double)i + 0.5) * d;
}

/*!
 * beta   ^    (T,3)
 *        |-----------
 *        |          |
 *  (L,0) |          | (R,1)
 *        |          |
 *        --------------> alpha
 *            (B,2)
 * Convert angle on a face to a fractional center index (for 1-D interp).
 * beta: varies in Y (L/R)
 * alpha: varies in X (B/T)
 * Returns u in "cell-center units": 0.0 ~ center 0, 1.0 ~ center 1, ... (N-1)
 */
inline double cs_angle_to_center_u(double angle, int N) {
  double d = M_PI / (2.0 * (double)N);
  // centers: angle = -pi/4 + (i+0.5)*d  => i = (angle + pi/4)/d - 0.5
  return (angle + M_PI / 4.0) / d - 0.5;
}

/*!
 * Side indices: {L:0, R:1, B:2, T:3}
 * Linear indexing of : [face][side][depth-1][j], where:
 * face <- [0..5], side <- [0..3]
 * depth <- [1..nghost], j <- [0..N-1]
 */
inline size_t cs_usrc_index(int face, int side, int depth, int j, int N,
                            int nghost) {
  const size_t S = 4;
  return ((size_t)face * S * (size_t)nghost * (size_t)N) +
         ((size_t)(side - SIDE_L) * (size_t)nghost * (size_t)N) +
         ((size_t)(depth - 1) * (size_t)N) + (size_t)j;
}

/*!
 *
 * Face names and global cartesian coordinates
 * -------------------------------------------
 *
 *       +z
 *       ^
 *       |
 *       |----> +y
 *      /
 *  +x /
 *
 * Local face coordinates
 * ----------------------
 *
 *         (T,3)          beta
 *        |-----|         ^
 *  (L,0) |  X  | (R,1)   |
 *        |-----|         |----> alpha
 *         (B,2)
 *
 * Side numbering
 * --------------
 *   Side_L = 0
 *   Side_R = 1
 *   Side_B = 2
 *   Side_T = 3
 *
 * Project unit vector to (alpha,bea) on a chosen face (no face selection).
 * Inverse of the above patterns: for +X, a=Y/X, b=Z/X, then alpha=atan(a), etc.
 * Assumes the vector is visible on that face (denominator sign consistent).
 */
void cs_xyz_to_ab(char const *face, Vec3 v, double *alpha, double *beta);

/*!
 * From local (alpha,beta) to unit vector on S^2 for a given face.
 * Gnomonic Equiangular: a = tan(alpha), b = tan(beta).
 * For +X face: (X,Y,Z) -> (1, a, b); normalize.
 */
Vec3 cs_ab_to_xyz(char const *face, double alpha, double beta);

/*!
 * Ghost cell centers just outside the panel.
 * side: SIDE_L,SIDE_R,SIDE_B,SIDE_T (left/right/bottom/top)
 * N: cells per dim (px==py==N)
 * j_along: along-edge index (0..N-1) (varies in y for L/R, in x for B/T)
 * o_depth: ghost depth (1..nghost)
 * Out: (alpha_t, beta_t) on the target face (ghost location angles)
 */
void cs_equ_ghost_center(int side, int N, int j_along, int o_depth,
                         double *alpha_t, double *beta_t);

/*!
 * map target ghost -> source 1-D coordinate
 * Inputs:
 *   face_t:  target face id
 *   side_t:  target side (SIDE_L/R/B/T)
 *   N:       cells per dim (px==py==N)
 *   j_along: target along-edge index in [0..N-1]
 *   depth_o: ghost depth (1..nghost)
 *
 * Outputs:
 *   *face_s, *side_s: source face and side (from CS_FACE_EDGES)
 *   *alpha_s, *beta_s: source angles of the mapped ghost point (optional debug)
 *
 * Return:
 *   u_src:  fractional index along the source edge line (for 1-D interp)
 *
 * Usage: sample your source data along the edge-aligned interior line
 *        (the row/col adjacent to side_s) at position u_src with 1-D
 * interpolation.
 */
double cs_target_ghost_to_source_u(int face_t, int side_t, int N, int j_along,
                                   int depth_o, int *face_s, int *side_s,
                                   double *alpha_s, double *beta_s);

/*!
 * Build full ghost->source interpolation table for all faces & edges.
 * Inputs:
 *   N       : cells per dimension on each face (px==py==N)
 *   nghost  : number of ghost layers to fill (>=1)
 *
 * Return:
 *   usrc : source 1D location with length 6 * 4 * nghost * N elements
 *          (use cs_usrc_index(...) to access)
 */
std::vector<double> cs_build_ghost_usrc(int N, int nghost,
                                        int apply_rev_flag = false);

}  // namespace snap
