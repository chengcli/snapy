// C/C++
#include <cstdio>

// snap
#include <snap/coord/cubed_sphere_utils.hpp>
#include <snap/layout/cubed_sphere_layout.hpp>

void run_demo(snap::CubedSphereLayout const &cs, int face, int rx, int ry) {
  printf(
      "Demo cubed-sphere Z-order connectivity pxy=%u face=%d (rx,ry)=(%u,%u)\n",
      cs.get_procs(), face, rx, ry);

  int g_self = cs.neighbor_rank(face, rx, ry, 0, 0);
  int g_left = cs.neighbor_rank(face, rx, ry, -1, 0);
  int g_right = cs.neighbor_rank(face, rx, ry, 1, 0);
  int g_down = cs.neighbor_rank(face, rx, ry, 0, -1);
  int g_up = cs.neighbor_rank(face, rx, ry, 0, 1);
  int g_ul = cs.neighbor_rank(face, rx, ry, -1, 1); /* corner */
  int g_dr = cs.neighbor_rank(face, rx, ry, 1, -1); /* corner */

  printf("self=%d L=%d R=%d D=%d U=%d UL=%d DR=%d\n", g_self, g_left, g_right,
         g_down, g_up, g_ul, g_dr);
}

void run_ghost(int nxy, int nghost) {
  auto gmap = snap::cs_build_ghost_map(nxy, nghost, false);

  for (int face = 0; face < 6; ++face) {
    printf("\nface %d ghost map:\n", face);
    for (int side = snap::SIDE_L; side <= snap::SIDE_T; ++side) {
      const char *sname = (side == snap::SIDE_L)   ? "L"
                          : (side == snap::SIDE_R) ? "R"
                          : (side == snap::SIDE_B) ? "B"
                          : (side == snap::SIDE_T) ? "T"
                                                   : "?";
      printf(" side %s:\n", sname);
      for (int depth = 1; depth <= nghost; ++depth) {
        for (int j = 0; j < nxy; ++j) {
          size_t idx = snap::cs_gmap_index(face, side, depth, j, nxy, nghost);
          auto gm = gmap[idx];
          printf("  d=%d j=%2d: u_src=%7.3f (alpha_s=%7.3f beta_s=%7.3f)\n",
                 depth, j, gm.u_src, gm.alpha_s, gm.beta_s);
        }
      }
    }
  }
}

int main(void) {
  int pxy = 4;
  snap::CubedSphereLayout cs(pxy);

  for (int n = 0; n < 6; ++n) {
    printf("\nface %d tests:\n", n);
    run_demo(cs, n, 0, 0);
    run_demo(cs, n, 0, 1);
    run_demo(cs, n, 1, 0);
    run_demo(cs, n, 1, 1);
  }

  run_ghost(6, 3);
  return 0;
}
