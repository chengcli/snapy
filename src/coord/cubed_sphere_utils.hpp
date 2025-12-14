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
 *   usrc    : array of length nghost * N, where usrc[d*N + j]
 *             is the source coordinate along the edge line for target
 *             ghost cell
 */
void cs_build_ghost_usrc(double *usrc, int N, int nghost, int face_t = 0,
                         int side_t = 0);

std::pair<torch::Tensor, torch::Tensor> cs_ab_to_lonlat(char const *face,
                                                        torch::Tensor alpha,
                                                        torch::Tensor beta);

//! \brief Transform cartesian velocities to contravariant velocities
/*!
 * Common variables
 * \f{eqnarray*}{
 *  x & = & \tan(\alpha) \\
 *  y & = & \tan(\beta) \\
 *  \delta & = & \sqrt{x^2 + y^2 + 1} \\
 *  C & = & \sqrt{1 + x^2} \\
 *  D & = & \sqrt{1 + y^2}
 * \f}
 *
 * Contravariant basis:
 * \f{eqnarray*}{
 *  \mathbf{b}^1 & = & \frac{1}{\delta} (x, y, 1)^T \\
 *  \mathbf{b}^2 & = & \frac{D}{\delta} (1, 0, -x)^T \\
 *  \mathbf{b}^3 & = & \frac{C}{\delta} (0, 1, -y)^T
 * \f}
 *
 * Velocity vector expressed using cartesian basis:
 * \f[
 *  \mathbf{v} = v_x \hat{\mathbf{x}} + v_y \hat{\mathbf{y}} + v_z
 * \hat{\mathbf{z}}
 * \f]
 *
 * Contravariant velocity components:
 * \f{eqnarray*}{
 *  v^1 & = & \mathbf{v} \cdot \mathbf{b}^1 = \frac{1}{\delta} (x v_x + y v_y +
 * v_z) \\
 *  v^2 & = & \mathbf{v} \cdot \mathbf{b}^2 = \frac{D}{\delta} (v_x - x v_z) \\
 *  v^3 & = & \mathbf{v} \cdot \mathbf{b}^3 = \frac{C}{\delta} (v_y - y v_z)
 * \f}
 */
void cs_cart_to_contra_(torch::Tensor const &vel, torch::Tensor alpha,
                        torch::Tensor beta);

//! \brief Transform contravariant velocities to cartesian velocities
/*!
 * Common variables
 * \f{eqnarray*}{
 *  x & = & \tan(\alpha) \\
 *  y & = & \tan(\beta) \\
 *  \delta & = & \sqrt{x^2 + y^2 + 1} \\
 *  C & = & \sqrt{1 + x^2} \\
 *  D & = & \sqrt{1 + y^2}
 * \f}
 *
 * Covariant basis:
 * \f{eqnarray*}{
 *  \mathbf{b}_1 & = & \frac{1}{\delta} (x, y, 1)^T \\
 *  \mathbf{b}_2 & = & \frac{1}{D \delta} (D^2, -x y, -x)^T \\
 *  \mathbf{b}_3 & = & \frac{1}{C \delta} (-x y, C^2, -y)^T
 * \f}
 *
 * Velocity vector expressed using covariant basis:
 * \f[
 *  \mathbf{v} = v^1 \mathbf{b}_1 + v^2 \mathbf{b}_2 + v^3 \mathbf{b}_3
 * \f]
 *
 * Cartesian velocity components:
 * \f{eqnarray*}{
 *  v_x & = & \mathbf{v} \cdot \hat{\mathbf{x}} = \frac{1}{\delta} (v^1 x + v^2
 * D - \frac{v^3 x y}{C}) \\
 *  v_y & = & \mathbf{v} \cdot \hat{\mathbf{y}} = \frac{1}{\delta} (v^1 y + v^3
 * C - \frac{v^2 x y}{D}) \\ v_z & = & \mathbf{v} \cdot \hat{\mathbf{z}} =
 * \frac{1}{\delta} (v^1 - \frac{v^2 x}{D} - \frac{v^3 y}{C})
 * \f}
 */
void cs_contra_to_cart_(torch::Tensor const &vel, torch::Tensor alpha,
                        torch::Tensor beta);

}  // namespace snap
