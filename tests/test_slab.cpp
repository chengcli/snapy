// C/C++
#include <cstdio>

// snap
#include <snap/layout/layout.hpp>

int main(void) {
  snap::LayoutOptions op;
  op.px(4);
  op.py(3);
  op.periodic_x(true);
  op.periodic_y(false);

  printf("2D demo %ux%u (periodic_x=%d periodic_y=%d)\n", op.px(), op.py(),
         op.periodic_x(), op.periodic_y());

  snap::SlabLayoutImpl sl(op);

  for (int ry = 0; ry < op.py(); ++ry)
    for (int rx = 0; rx < op.px(); ++rx) {
      int left = sl.neighbor_rank(rx, ry, 0, -1, 0, 0);
      int right = sl.neighbor_rank(rx, ry, 0, 1, 0, 0);
      int up = sl.neighbor_rank(rx, ry, 0, 0, 1, 0);
      int down = sl.neighbor_rank(rx, ry, 0, 0, -1, 0);
      int r = sl.rank_of(rx, ry, 0);
      printf("rank %2d @ (y=%d,x=%d): L=%2d R=%2d U=%2d D=%2d\n", r, ry, rx,
             left, right, up, down);
    }

  return 0;
}
