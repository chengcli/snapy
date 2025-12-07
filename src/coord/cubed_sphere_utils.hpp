#pragma once

// C/C++
#include <cmath>
#include <cstdint>
#include <vector>

// snap
#include <snap/layout/cubed_sphere_layout.hpp>

namespace snap {

enum { AX_X = 0, AX_Y = 1 };

struct Vec3 {
  double x, y, z;
};

inline Vec3 unit_vec3(double x, double y, double z) {
  double norm = std::sqrt(x * x + y * y + z * z);
  return Vec3{x / norm, y / norm, z / norm};
}

//! L/R vary in Y; B/T vary in X
inline int cs_side_axis(int side) {
  return (side == SIDE_L || side == SIDE_R) ? AX_Y : AX_X;
}

//! outward normal: left/bottom = -1; right/top = +1
inline int cs_side_sign(int side) {
  return (side == SIDE_R || side == SIDE_T) ? +1 : -1;
}

/*!
 * Equiangular centers & ghost centers
 * cell centers: [-pi/4, pi/4], d = pi/(2N), center i in [0..N-1]
 */
inline double cs_equ_center(int N, int i) {
  double d = M_PI / (2.0 * (double)N);
  return -M_PI / 4.0 + ((double)i + 0.5) * d;
}

/*!
 * Ghost cell centers just outside the panel.
 * side: SIDE_L,SIDE_R,SIDE_B,SIDE_T (left/right/bottom/top)
 * N: cells per dim (px==py==N)
 * j_along: along-edge index (0..N-1) (varies in y for L/R, in x for B/T)
 * o_depth: ghost depth (1..nghost)
 * Out: (alpha_t, beta_t) on the target face (ghost location angles)
 */
inline void cs_equ_ghost_center(int side, int N, int j_along, int depth,
                                double *alpha_t, double *beta_t) {
  double d = M_PI / (2.0 * (double)N);
  int sgn = cs_side_sign(side);  // -1 for L/B, +1 for R/T
  int ax = cs_side_axis(side);   // AX_Y: edge varies in beta; AX_X: in alpha
  double along = cs_equ_center(N, j_along);
  double perp = sgn * (M_PI / 4.0 + ((double)depth - 0.5) * d);

  if (ax == AX_Y) {  // L/R: alpha outwards, beta along
    *alpha_t = perp;
    *beta_t = along;
  } else {  // B/T: alpha along, beta outwards
    *alpha_t = along;
    *beta_t = perp;
  }
}

/*!
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
 * Build full ghost->source interpolation table for all faces & edges.
 * Inputs:
 *   N       : cells per dimension on each face (px==py==N)
 *   nghost  : number of ghost layers to fill (>=1)
 *   face_t  : target face index [0..5]
 *   side_t  : target side index (SIDE_L/RIGHT/BOTTOM/TOP)
 *
 * Output:
 *   usrc    : array of length nghost * N, where usrc[d*N + j] is the
 *             source coordinate along the edge line for target ghost cell
 */
void cs_build_ghost_usrc(double *usrc, int N, int nghost, int face_t = 0,
                         int side_t = 0);

}  // namespace snap
