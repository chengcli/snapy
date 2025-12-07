// C/C++
#include <cmath>
#include <cstdint>
#include <cstring>

// snap
#include "cubed_sphere_utils.hpp"

namespace snap {

/*!
 * beta   ^ (T,3)
 *        |------
 *  (L,0) |  X  | (R,1)
 *        --------> alpha
 *            (B,2)
 * Convert angle on a face to a fractional center index (for 1-D interp).
 * beta: varies in Y (L/R)
 * alpha: varies in X (B/T)
 * Returns u in "cell-center units": 0.0 ~ center 0, 1.0 ~ center 1, ... (N-1)
 */
static inline double cs_angle_to_center_u(double angle, int N) {
  double d = M_PI / (2.0 * (double)N);
  // centers: angle = -pi/4 + (i+0.5)*d  => i = (angle + pi/4)/d - 0.5
  return (angle + M_PI / 4.0) / d - 0.5;
}

Vec3 cs_ab_to_xyz(char const *face, double alpha, double beta) {
  double a = tan(alpha), b = tan(beta);
  if (strcmp(face, "+X") == 0)
    return unit_vec3(1.0, a, b);
  else if (strcmp(face, "-X") == 0)
    return unit_vec3(-1.0, -a, b);
  else if (strcmp(face, "+Y") == 0)
    return unit_vec3(-a, 1.0, b);
  else if (strcmp(face, "-Y") == 0)
    return unit_vec3(a, -1.0, b);
  else if (strcmp(face, "+Z") == 0)
    return unit_vec3(-b, a, 1.0);
  else if (strcmp(face, "-Z") == 0)
    return unit_vec3(b, a, -1.0);
  else
    throw std::runtime_error("cs_ab_to_xyz: invalid face name");
}

void cs_xyz_to_ab(char const *face, Vec3 v, double *alpha, double *beta) {
  if (strcmp(face, "+X") == 0) {
    *alpha = atan2(v.y, v.x);
    *beta = atan2(v.z, v.x);
  } else if (strcmp(face, "-X") == 0) {
    *alpha = atan2(-v.y, -v.x);
    *beta = atan2(v.z, -v.x);
  } else if (strcmp(face, "+Y") == 0) {
    *alpha = atan2(-v.x, v.y);
    *beta = atan2(v.z, v.y);
  } else if (strcmp(face, "-Y") == 0) {
    *alpha = atan2(v.x, -v.y);
    *beta = atan2(v.z, -v.y);
  } else if (strcmp(face, "+Z") == 0) {
    *alpha = atan2(v.y, v.z);
    *beta = atan2(-v.x, v.z);
  } else if (strcmp(face, "-Z") == 0) {
    *alpha = atan2(v.y, -v.z);
    *beta = atan2(v.x, -v.z);
  } else {
    throw std::runtime_error("cs_xyz_to_ab: invalid face name");
  }
}

void cs_build_ghost_usrc(double *usrc, int N, int nghost, int face_t,
                         int side_t) {
  const CSEdge emap = CS_FACE_EDGES[face_t][side_t];

  // which angle varies on target face
  const int src_ax = cs_side_axis(emap.nside);
  for (int depth = 1; depth <= nghost; ++depth) {
    for (int j = 0; j < N; ++j) {
      // 1) Target ghost center
      double alpha_t, beta_t;
      cs_equ_ghost_center(side_t, N, j, depth, &alpha_t, &beta_t);

      // 2) To cartesian
      Vec3 v = cs_ab_to_xyz(CS_FACE_NAMES[face_t], alpha_t, beta_t);

      // 3) Re-express on the source face from connectivity
      double alpha_s, beta_s;
      cs_xyz_to_ab(CS_FACE_NAMES[emap.nface], v, &alpha_s, &beta_s);

      // 4) Fractional abscissa along source edge’s interior line
      double u_src =
          cs_angle_to_center_u((src_ax == AX_Y) ? beta_s : alpha_s, N);

      // 5) Flip per connectivity flag
      if (emap.rev) u_src = (double)(N - 1) - u_src;

      // 6) Write out
      usrc[(depth - 1) * N + j] = u_src;
    }
  }
}

}  // namespace snap
