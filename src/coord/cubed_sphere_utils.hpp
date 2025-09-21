// C/C++
#include <cmath>

namespace snap {

/* One ghost cell mapping (target ghost -> source 1-D location) */
struct CSGhostMap {
  int64_t face_s;  /* source face id */
  int64_t side_s;  /* source side id (SIDE_L/R/B/T) */
  int64_t j_along; /* along-edge index on target (0..N-1) */
  int64_t depth;   /* ghost depth (1..nghost) */

  double u_src; /* fractional index along the *source* edge interior line */
  double xi_s;  /* optional: source face angle xi (debug/validation) */
  double eta_s; /* optional: source face angle eta (debug/validation) */
} CSGhostMap;

/* ------------- Equiangular centers & ghost centers -------------- */
/* cell centers: [-pi/4, pi/4], Δ = pi/(2N), center i in [0..N-1] */
inline double cs_equ_center(int N, int i) {
  double d = M_PI / (2.0 * (double)N);
  return -M_PI / 4.0 + ((double)i + 0.5) * d;
}

/* ------------- map target ghost -> source 1-D coordinate ----
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
