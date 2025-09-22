// C/C++
#include <cmath>
#include <cstdint>
#include <cstring>

// snap
#include "cubed_sphere_utils.hpp"

namespace snap {

extern const CSEdge CS_FACE_EDGES[6][4];
extern const char face_names[6][3];

void cs_equ_ghost_center(int side, int N, int j_along, int o_depth,
                         double *alpha_t, double *beta_t) {
  double d = M_PI / (2.0 * (double)N);
  switch (side) {
    case SIDE_L: /* alpha just outside left */
      *alpha_t = -M_PI / 4.0 - ((double)o_depth - 0.5) * d;
      *beta_t = cs_equ_center(N, j_along);
      break;
    case SIDE_R: /* alpha just outside right */
      *alpha_t = M_PI / 4.0 + ((double)o_depth - 0.5) * d;
      *beta_t = cs_equ_center(N, j_along);
      break;
    case SIDE_B: /* beta just outside bottom */
      *beta_t = -M_PI / 4.0 - ((double)o_depth - 0.5) * d;
      *alpha_t = cs_equ_center(N, j_along);
      break;
    case SIDE_T: /* beta just outside top */
      *beta_t = M_PI / 4.0 + ((double)o_depth - 0.5) * d;
      *alpha_t = cs_equ_center(N, j_along);
      break;
    default:
      *alpha_t = 0.0;
      *beta_t = 0.0; /* invalid */
  }
}

/* ---------- 2) Face <-> sphere transforms (equiangular gnomonic) --- */
/* We number faces: 0:+X, 1:+Y, 2:-X, 3:-Y, 4:+Z, 5:-Z (edit to match yours) */

static inline Vec3 vnorm3(double x, double y, double z) {
  double n = sqrt(x * x + y * y + z * z);
  Vec3 v = {x / n, y / n, z / n};
  return v;
}

Vec3 cs_ab_to_xyz(char const *face, double alpha, double beta) {
  double a = tan(alpha), b = tan(beta);
  if (strcmp(face, "+X") == 0)
    return vnorm3(1.0, a, b);
  else if (strcmp(face, "-X") == 0)
    return vnorm3(-1.0, -a, b);
  else if (strcmp(face, "+Y") == 0)
    return vnorm3(-a, 1.0, b);
  else if (strcmp(face, "-Y") == 0)
    return vnorm3(a, -1.0, b);
  else if (strcmp(face, "+Z") == 0)
    return vnorm3(-b, a, 1.0);
  else if (strcmp(face, "-Z") == 0)
    return vnorm3(b, a, -1.0);
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

double cs_target_ghost_to_source_u(int face_t, int side_t, int N, int j_along,
                                   int depth_o, int *face_s, int *side_s,
                                   double *alpha_s, double *beta_s) {
  double u_src;

  /* 1) Get which neighbor face/side we land on from your connectivity */
  const CSEdge emap = CS_FACE_EDGES[face_t][side_t];
  *face_s = emap.nface;
  *side_s = emap.nside;

  /* 2) Compute the target ghost center angles on the target face */
  double alpha_t, beta_t;
  cs_equ_ghost_center(side_t, N, j_along, depth_o, &alpha_t, &beta_t);

  /* 3) Map that ghost point to the sphere */
  Vec3 v = cs_ab_to_xyz(face_names[face_t], alpha_t, beta_t);

  /* 4) Re-express the same point in source face local coords */
  cs_xyz_to_ab(face_names[*face_s], v, alpha_s, beta_s);

  /* 5) Build the along-edge 1-D interpolation coordinate on the source edge
   *    L/R sides vary in Y (beta); B/T sides vary in X (alpha).
   *    We do NOT manually flip for orientation; the spherical projection
   *    + chosen inverse mapping handles the correct geometric direction.
   */
  switch (*side_s) {
    case SIDE_L:
    case SIDE_R:
      u_src = cs_angle_to_center_u(*beta_s, N); /* varies in Y */
      break;
    case SIDE_B:
    case SIDE_T:
      u_src = cs_angle_to_center_u(*alpha_s, N); /* varies in X */
      break;
    default:
      throw std::runtime_error("cs_target_ghost_to_source_u: invalid side");
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
  return u_src;
}

std::vector<double> cs_build_ghost_usrc(int N, int nghost, int apply_rev_flag) {
  std::vector<double> usrc(6 * 4 * nghost * N);

  for (int face_t = 0; face_t < 6; ++face_t) {
    for (int side_t = SIDE_L; side_t <= SIDE_T; ++side_t) {
      const CSEdge emap =
          CS_FACE_EDGES[face_t][side_t]; /* neighbor face/side + rev flag */
      for (int depth = 1; depth <= nghost; ++depth) {
        for (int j = 0; j < N; ++j) {
          /* 1) Target ghost center angles on target face */
          double alpha_t, beta_t; /* angular coordinates */
          cs_equ_ghost_center(side_t, N, j, depth, &alpha_t, &beta_t);

          /* 2) Map to the sphere */
          Vec3 v = cs_ab_to_xyz(face_names[face_t], alpha_t, beta_t);

          /* 3) Re-express on the source face from connectivity */
          double alpha_s, beta_s;
          cs_xyz_to_ab(face_names[emap.nface], v, &alpha_s, &beta_s);

          /* 4) Build 1-D fractional index along the source edge interior line
           */
          double u_src;
          switch (emap.nside) {
            case SIDE_L:
            case SIDE_R:
              u_src = cs_angle_to_center_u(beta_s, N); /* varies along Y */
              break;
            case SIDE_B:
            case SIDE_T:
              u_src = cs_angle_to_center_u(alpha_s, N); /* varies along X */
              break;
            default:
              u_src = 0.0; /* should not happen */
          }

          /* 5) Optional index-based reversal (discrete orientation) */
          if (apply_rev_flag && emap.rev) {
            u_src = (double)(N - 1) - u_src;
          }

          /* 6) Write out */
          size_t idx = cs_usrc_index(face_t, side_t, depth, j, N, nghost);
          usrc[idx] = u_src;
        }
      }
    }
  }

  return usrc;
}

}  // namespace snap
