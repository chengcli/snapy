// C/C++
#include <cstdio>

// canoe
#include <snap/layout/slab_layout.hpp>

int main(void) {
  int px = 4;
  int py = 3;
  bool periodic_x = true;
  bool periodic_y = false;

  printf("2D demo %ux%u (periodic_x=%d periodic_y=%d)\n", px, py, periodic_x,
         periodic_y);

  canoe::SlabLayout sl(px, py, periodic_x, periodic_y);

  for (int ry = 0; ry < py; ++ry)
    for (int rx = 0; rx < px; ++rx) {
      int left = sl.neighbor_rank(rx, ry, -1, 0);
      int right = sl.neighbor_rank(rx, ry, 1, 0);
      int up = sl.neighbor_rank(rx, ry, 0, 1);
      int down = sl.neighbor_rank(rx, ry, 0, -1);
      int r = sl.rank_of(rx, ry);
      printf("rank %2d @ (y=%d,x=%d): L=%2d R=%2d U=%2d D=%2d\n", r, ry, rx,
             left, right, up, down);
    }

  return 0;
}
