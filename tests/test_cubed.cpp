// C/C++
#include <cstdio>

// canoe
#include <snap/layout/cubed_layout.hpp>

int main(void) {
  int px = 2;
  int py = 2;
  int pz = 2;
  bool periodic_x = true;
  bool periodic_y = true;
  bool periodic_z = true;

  printf("3D demo %ux%u×%u (periodic=%d,%d,%d)\n", px, py, pz, periodic_x,
         periodic_y, periodic_z);

  canoe::CubedLayout cl(px, py, pz, periodic_x, periodic_y, periodic_z);

  for (int rx = 0; rx < px; ++rx)
    for (int ry = 0; ry < py; ++ry)
      for (int rz = 0; rz < pz; ++rz) {
        int xp = cl.neighbor_rank(rx, ry, rz, 1, 0, 0);
        int xm = cl.neighbor_rank(rx, ry, rz, -1, 0, 0);
        int yp = cl.neighbor_rank(rx, ry, rz, 0, 1, 0);
        int ym = cl.neighbor_rank(rx, ry, rz, 0, -1, 0);
        int zp = cl.neighbor_rank(rx, ry, rz, 0, 0, 1);
        int zm = cl.neighbor_rank(rx, ry, rz, 0, 0, -1);
        int r = cl.rank_of(rx, ry, rz);
        printf(
            "rank %2d @ (z=%d,y=%d,x=%d): x-=%2d x+=%2d y-=%2d y+=%2d z-=%2d "
            "z+=%2d\n",
            r, rz, ry, rx, xm, xp, ym, yp, zm, zp);
      }

  return 0;
}
