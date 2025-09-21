// C/C++
#include <cmath.h>

namespace snap {

/* ------------- 1) Equiangular centers & ghost centers -------------- */
/* cell centers: [-pi/4, pi/4], Δ = pi/(2N), center i in [0..N-1] */
template <typename T>
T cs_equ_center(int N, int i) {
  T d = M_PI / (2.0 * (T)N);
  return -M_PI / 4.0 + ((T)i + 0.5) * d;
}

/* Ghost cell centers just outside the panel.
 * side: SIDE_L,SIDE_R,SIDE_B,SIDE_T (left/right/bottom/top)
 * N: cells per dim (px==py==N)
 * j_along: along-edge index (0..N-1) (varies in y for L/R, in x for B/T)
 * o_depth: ghost depth (1..nghost)
 * Out: (xi_t, eta_t) on the target face (ghost location angles)
 */
template <typename T>
void cs_equ_ghost_center(int side, int N, int j_along, int o_depth, T *xi_t,
                         T *eta_t) {
  T d = M_PI / (2.0 * (T)N);
  switch (side) {
    case SIDE_L: /* xi just outside left */
      *xi_t = -M_PI / 4.0 - ((T)o_depth - 0.5) * d;
      *eta_t = cs_equ_center(N, j_along);
      break;
    case SIDE_R: /* xi just outside right */
      *xi_t = M_PI / 4.0 + ((T)o_depth - 0.5) * d;
      *eta_t = cs_equ_center(N, j_along);
      break;
    case SIDE_B: /* eta just outside bottom */
      *eta_t = -M_PI / 4.0 - ((T)o_depth - 0.5) * d;
      *xi_t = cs_equ_center(N, j_along);
      break;
    case SIDE_T: /* eta just outside top */
      *eta_t = M_PI / 4.0 + ((T)o_depth - 0.5) * d;
      *xi_t = cs_equ_center(N, j_along);
      break;
    default:
      *xi_t = 0.0;
      *eta_t = 0.0; /* invalid */
  }
}

/* ---------- 2) Face <-> sphere transforms (equiangular gnomonic) --- */
/* We number faces: 0:+X, 1:+Y, 2:-X, 3:-Y, 4:+Z, 5:-Z (edit to match yours) */

template <typename T>
struct Vec3 {
  T x, y, z;
}

template <typename T>
Vec3<T> vnorm3(T x, T y, T z) {
  T n = sqrt(x * x + y * y + z * z);
  Vec3 v = {x / n, y / n, z / n};
  return v;
}

/* From local (xi,eta) to unit vector on S^2 for a given face.
 * Equiangular gnomonic: a = tan(xi), b = tan(eta).
 * For +X face: (X,Y,Z) ∝ (1, a, b); normalize.
 */
template <typename T>
Vec3<T> cs_face_to_vec(int face, T xi, T eta) {
  T a = tan(xi), b = tan(eta);
  switch (face) {
    case 0: /* +X */
      return vnorm3(1.0, a, b);
    case 1: /* +Y */
      return vnorm3(-a, 1.0, b);
    case 2: /* -X */
      return vnorm3(-1.0, -a, -b);
    case 3: /* -Y */
      return vnorm3(a, -1.0, -b);
    case 4: /* +Z */
      return vnorm3(-b, a, 1.0);
    case 5: /* -Z */
      return vnorm3(b, -a, -1.0);
    default:
      return vnorm3(1, 0, 0);
  }
}

/* Project unit vector to (xi,eta) on a chosen face (no face selection).
 * Inverse of the above patterns: for +X, a=Y/X, b=Z/X, then xi=atan(a), etc.
 * Assumes the vector is visible on that face (denominator sign consistent).
 */
template <typename T>
void cs_vec_to_face_coords(int face, Vec3 v, T *xi, T *eta) {
  switch (face) {
    case 0: /* +X */
      *xi = atan2(v.y, v.x);
      *eta = atan2(v.z, v.x);
      break;
    case 1: /* +Y */
      *xi = atan2(-v.x, v.y);
      *eta = atan2(v.z, v.y);
      break;
    case 2: /* -X */
      *xi = atan2(-v.y, -v.x);
      *eta = atan2(-v.z, -v.x);
      break;
    case 3: /* -Y */
      *xi = atan2(v.x, -v.y);
      *eta = atan2(-v.z, -v.y);
      break;
    case 4: /* +Z */
      *xi = atan2(v.y, v.z);
      *eta = atan2(-v.x, v.z);
      break;
    case 5: /* -Z */
      *xi = atan2(-v.y, -v.z);
      *eta = atan2(v.x, -v.z);
      break;
    default:
      *xi = 0.0;
      *eta = 0.0;
      break;
  }
}

/* Convert angle on a face to a fractional center index (for 1-D interp).
 * If the line varies in Y (L/R sides): use eta. If varies in X (B/T): use xi.
 * Returns u in "cell-center units": 0.0 ~ center 0, 1.0 ~ center 1, ... (N-1)
 */
template <typename T>
T cs_angle_to_center_u(T angle, int N) {
  T d = M_PI / (2.0 * (T)N);
  /* centers: angle = -pi/4 + (i+0.5)*d  => i = (angle + pi/4)/d - 0.5 */
  return (angle + M_PI / 4.0) / d - 0.5;
}

/* ------------- 3) Main: map target ghost -> source 1-D coordinate ----
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
template <typename T>
void cs_target_ghost_to_source_u(int face_t, int side_t, int N, int j_along,
                                 int depth_o, int *face_s, int *side_s,
                                 T *u_src, T *xi_s, T *eta_s) {
  /* 1) Get which neighbor face/side we land on from your connectivity */
  const CSEdge emap = CS_FACE_EDGES[face_t][side_t];
  *face_s = emap.nface;
  *side_s = emap.nside;

  /* 2) Compute the target ghost center angles on the target face */
  T xi_t, eta_t;
  cs_equ_ghost_center(side_t, N, j_along, depth_o, &xi_t, &eta_t);

  /* 3) Map that ghost point to the sphere */
  Vec3 v = cs_face_to_vec(face_t, xi_t, eta_t);

  /* 4) Re-express the same point in source face local coords */
  cs_vec_to_face_coords(*face_s, v, xi_s, eta_s);

  /* 5) Build the along-edge 1-D interpolation coordinate on the source edge
   *    L/R sides vary in Y (eta); B/T sides vary in X (xi).
   *    We do NOT manually flip for orientation; the spherical projection
   *    + chosen inverse mapping handles the correct geometric direction.
   */
  switch (*side_s) {
    case SIDE_L:
    case SIDE_R:
      *u_src = cs_angle_to_center_u(*eta_s, N); /* varies in Y */
      break;
    case SIDE_B:
    case SIDE_T:
      *u_src = cs_angle_to_center_u(*xi_s, N); /* varies in X */
      break;
    default:
      *u_src = 0.0;
  }

  /* Optional: if you want to explicitly enforce the orientation flag (emap.rev)
   * on the 1-D coordinate (purely index-based reversal), uncomment:
   *
   * if (emap.rev) {
   *     *u_src = ( (double)(N-1) ) - *u_src;
   * }
   *
   * In many setups the geometric transform already encodes the "flip".
   * Keep this off unless your validation shows a reversed ordering.
   */
}

}  // namespace snap
