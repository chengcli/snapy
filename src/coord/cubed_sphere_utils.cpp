// C/C++
#include <cmath>

// snap
#include "cubed_sphere_utils.hpp"

namespace snap {

extern const CSEdge CS_FACE_EDGES[6][4];

void cs_equ_ghost_center(int side, int N, int j_along, int o_depth,
                         double *xi_t, double *eta_t) {
  double d = M_PI / (2.0 * (double)N);
  switch (side) {
    case SIDE_L: /* xi just outside left */
      *xi_t = -M_PI / 4.0 - ((double)o_depth - 0.5) * d;
      *eta_t = cs_equ_center(N, j_along);
      break;
    case SIDE_R: /* xi just outside right */
      *xi_t = M_PI / 4.0 + ((double)o_depth - 0.5) * d;
      *eta_t = cs_equ_center(N, j_along);
      break;
    case SIDE_B: /* eta just outside bottom */
      *eta_t = -M_PI / 4.0 - ((double)o_depth - 0.5) * d;
      *xi_t = cs_equ_center(N, j_along);
      break;
    case SIDE_T: /* eta just outside top */
      *eta_t = M_PI / 4.0 + ((double)o_depth - 0.5) * d;
      *xi_t = cs_equ_center(N, j_along);
      break;
    default:
      *xi_t = 0.0;
      *eta_t = 0.0; /* invalid */
  }
}

/* ---------- 2) Face <-> sphere transforms (equiangular gnomonic) --- */
/* We number faces: 0:+X, 1:+Y, 2:-X, 3:-Y, 4:+Z, 5:-Z (edit to match yours) */

static inline Vec3 vnorm3(double x, double y, double z) {
  double n = sqrt(x * x + y * y + z * z);
  Vec3 v = {x / n, y / n, z / n};
  return v;
}

Vec3 cs_face_to_vec(int face, double xi, double eta) {
  double a = tan(xi), b = tan(eta);
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

void cs_vec_to_face_coords(int face, Vec3 v, double *xi, double *eta) {
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

double cs_target_ghost_to_source_u(int face_t, int side_t, int N, int j_along,
                                   int depth_o, int *face_s, int *side_s,
                                   double *xi_s, double *eta_s) {
  double u_src;

  /* 1) Get which neighbor face/side we land on from your connectivity */
  const CSEdge emap = CS_FACE_EDGES[face_t][side_t];
  *face_s = emap.nface;
  *side_s = emap.nside;

  /* 2) Compute the target ghost center angles on the target face */
  double xi_t, eta_t;
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
      u_src = cs_angle_to_center_u(*eta_s, N); /* varies in Y */
      break;
    case SIDE_B:
    case SIDE_T:
      u_src = cs_angle_to_center_u(*xi_s, N); /* varies in X */
      break;
    default:
      u_src = 0.0;
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

std::vector<CSGhostMap> cs_build_ghost_map(int N, int nghost,
                                           int apply_rev_flag) {
  std::vector<CSGhostMap> gmap(6 * 4 * nghost * N);

  for (int face_t = 0; face_t < 6; ++face_t) {
    for (int side_t = SIDE_L; side_t <= SIDE_T; ++side_t) {
      const CSEdge emap =
          CS_FACE_EDGES[face_t][side_t]; /* neighbor face/side + rev flag */
      for (int depth = 1; depth <= nghost; ++depth) {
        for (int j = 0; j < N; ++j) {
          /* 1) Target ghost center angles on target face */
          double xi_t, eta_t;
          cs_equ_ghost_center(side_t, N, j, depth, &xi_t, &eta_t);

          /* 2) Map to the sphere */
          Vec3 v = cs_face_to_vec(face_t, xi_t, eta_t);

          /* 3) Re-express on the source face from connectivity */
          double xi_s, eta_s;
          cs_vec_to_face_coords(emap.nface, v, &xi_s, &eta_s);

          /* 4) Build 1-D fractional index along the source edge interior line
           */
          double u_src;
          switch (emap.nside) {
            case SIDE_L:
            case SIDE_R:
              u_src = cs_angle_to_center_u(eta_s, N); /* varies along Y */
              break;
            case SIDE_B:
            case SIDE_T:
              u_src = cs_angle_to_center_u(xi_s, N); /* varies along X */
              break;
            default:
              u_src = 0.0; /* should not happen */
          }

          /* 5) Optional index-based reversal (discrete orientation) */
          if (apply_rev_flag && emap.rev) {
            u_src = (double)(N - 1) - u_src;
          }

          /* 6) Write out */
          size_t idx = cs_gmap_index(face_t, side_t, depth, j, N, nghost);
          CSGhostMap &e = gmap[idx];
          e.u_src = u_src;
          e.xi_s = xi_s;
          e.eta_s = eta_s;
        }
      }
    }
  }

  return gmap;
}

}  // namespace snap
