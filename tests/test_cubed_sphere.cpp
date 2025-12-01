// C/C++
#include <cstdio>

// snap
#include <snap/layout/layout.hpp>

void run_demo(snap::CubedSphereLayoutImpl const &cs, int rx, int ry, int face) {
  printf(
      "Demo cubed-sphere Z-order connectivity pxy=%u face=%d (rx,ry)=(%u,%u)\n",
      cs.pxy(), face, rx, ry);

  int g_self = cs.neighbor_rank({rx, ry, face}, {0, 0, 0});
  int g_left = cs.neighbor_rank({rx, ry, face}, {-1, 0, 0});
  int g_right = cs.neighbor_rank({rx, ry, face}, {1, 0, 0});
  int g_down = cs.neighbor_rank({rx, ry, face}, {0, -1, 0});
  int g_up = cs.neighbor_rank({rx, ry, face}, {0, 1, 0});
  int g_ul = cs.neighbor_rank({rx, ry, face}, {-1, 1, 0}); /* corner */
  int g_dr = cs.neighbor_rank({rx, ry, face}, {1, -1, 0}); /* corner */

  printf("self=%d L=%d R=%d D=%d U=%d UL=%d DR=%d\n", g_self, g_left, g_right,
         g_down, g_up, g_ul, g_dr);
}

int main(void) {
  auto op = snap::LayoutOptionsImpl::create();
  op->px(4);
  op->py(4);

  snap::CubedSphereLayoutImpl cs(op);

  for (int n = 0; n < 6; ++n) {
    printf("\nface %d tests:\n", n);
    run_demo(cs, 0, 0, n);
    run_demo(cs, 0, 1, n);
    run_demo(cs, 1, 0, n);
    run_demo(cs, 1, 1, n);
  }
  return 0;
}
