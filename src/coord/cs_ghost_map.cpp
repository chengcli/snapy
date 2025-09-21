// snap
#include <snap/layout/cubed_sphere_layout.hpp>

#include "cubed_sphere_utils.hpp"

namespace snap {

extern const CSEdge CS_FACE_EDGES[6][4];

/* Indexing: [face][side][depth-1][j], where:
   face ∈ [0..5], side ∈ {SIDE_L..SIDE_T} (4 sides),
   depth ∈ [1..nghost], j ∈ [0..N-1] */
static inline size_t cs_gmap_index(int face, int side, int depth, int j, int N,
                                   int nghost) {
  const size_t S = 4; /* L,R,B,T */
  return ((size_t)face * S * (size_t)nghost * (size_t)N) +
         ((size_t)(side - SIDE_L) * (size_t)nghost * (size_t)N) +
         ((size_t)(depth - 1) * (size_t)N) + (size_t)j;
}

void cs_build_ghost_map_table(int N, int nghost, int apply_rev_flag,
                              CSGhostMap *gmap) {
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

          /* 5) Optional index-based reversal (purely discrete orientation) */
          if (apply_rev_flag && emap.rev) {
            u_src = (double)(N - 1) - u_src;
          }

          /* 6) Write out */
          size_t idx = cs_gmap_index(face_t, side_t, depth, j, N, nghost);
          CSGhostMap *e = gmap + idx;
          e->face_s = (int64_t)emap.nface;
          e->side_s = (int64_t)emap.nside;
          e->j_along = (int64_t)j;
          e->depth = (int64_t)depth;
          e->u_src = u_src;
          e->xi_s = xi_s;
          e->eta_s = eta_s;
        }
      }
    }
  }
}

}  // namespace snap
