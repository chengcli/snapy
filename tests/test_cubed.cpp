// C/C++
#include <cstdio>

// snap
#include <snap/layout/layout.hpp>

int main(void) {
  snap::LayoutOptions op;
  op.px(2);
  op.py(2);
  op.pz(2);
  op.periodic_x(true);
  op.periodic_y(true);
  op.periodic_y(true);

  printf("3D demo %ux%u×%u (periodic=%d,%d,%d)\n", op.px(), op.py(), op.pz(),
         op.periodic_x(), op.periodic_y(), op.periodic_z());

  snap::CubedLayoutImpl cl(op);

  for (int rx = 0; rx < op.px(); ++rx)
    for (int ry = 0; ry < op.py(); ++ry)
      for (int rz = 0; rz < op.pz(); ++rz) {
        int xp = cl.neighbor_rank({rx, ry, rz}, {1, 0, 0});
        int xm = cl.neighbor_rank({rx, ry, rz}, {-1, 0, 0});
        int yp = cl.neighbor_rank({rx, ry, rz}, {0, 1, 0});
        int ym = cl.neighbor_rank({rx, ry, rz}, {0, -1, 0});
        int zp = cl.neighbor_rank({rx, ry, rz}, {0, 0, 1});
        int zm = cl.neighbor_rank({rx, ry, rz}, {0, 0, -1});
        int r = cl.rank_of({rx, ry, rz});
        printf(
            "rank %2d @ (z=%d,y=%d,x=%d): x-=%2d x+=%2d y-=%2d y+=%2d z-=%2d "
            "z+=%2d\n",
            r, rz, ry, rx, xm, xp, ym, yp, zm, zp);
      }

  return 0;
}
