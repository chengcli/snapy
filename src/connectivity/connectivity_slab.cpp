// C/C++
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "connectivity.hpp"

/* ============================
 * Neighbor → Z-order rank (2D)
 * ============================
 * dx,dy ∈ {-1,0,1}. periodic flags control wrap; otherwise off-domain → -1.
 * (rx,ry) are THIS rank's coords in the process grid (not Morton code).
 */

int z_neighbor_rank2(int dx, int dy, uint32_t rx, uint32_t ry, uint32_t px,
                     uint32_t py, int periodic_x, int periodic_y,
                     const int *rank_of /* length py*px */
) {
  int nx = (int)rx + dx;
  int ny = (int)ry + dy;

  if (periodic_x) {
    if (nx < 0)
      nx += (int)px;
    else if (nx >= (int)px)
      nx -= (int)px;
  } else {
    if (nx < 0 || nx >= (int)px) return -1;
  }

  if (periodic_y) {
    if (ny < 0)
      ny += (int)py;
    else if (ny >= (int)py)
      ny -= (int)py;
  } else {
    if (ny < 0 || ny >= (int)py) return -1;
  }

  return rank_of[linear_index2(px, py, (uint32_t)ny, (uint32_t)nx)];
}

/* ============================
 * Neighbor → Z-order rank (3D)
 * ============================
 * dx,dy,dz ∈ {-1,0,1}. periodic flags control wrap; otherwise off-domain → -1.
 * (rx,ry,rz) are THIS rank's coords in the process grid (not Morton code).
 */

int z_neighbor_rank3(int dx, int dy, int dz, uint32_t rx, uint32_t ry,
                     uint32_t rz, uint32_t px, uint32_t py, uint32_t pz,
                     int periodic_x, int periodic_y, int periodic_z,
                     const int *rank_of /* length pz*py*px */
) {
  int nx = (int)rx + dx;
  int ny = (int)ry + dy;
  int nz = (int)rz + dz;

  if (periodic_x) {
    if (nx < 0)
      nx += (int)px;
    else if (nx >= (int)px)
      nx -= (int)px;
  } else {
    if (nx < 0 || nx >= (int)px) return -1;
  }

  if (periodic_y) {
    if (ny < 0)
      ny += (int)py;
    else if (ny >= (int)py)
      ny -= (int)py;
  } else {
    if (ny < 0 || ny >= (int)py) return -1;
  }

  if (periodic_z) {
    if (nz < 0)
      nz += (int)pz;
    else if (nz >= (int)pz)
      nz -= (int)pz;
  } else {
    if (nz < 0 || nz >= (int)pz) return -1;
  }

  return rank_of[linear_index3(px, py, (uint32_t)nz, (uint32_t)ny,
                               (uint32_t)nx)];
}

/* ============================
 * (Optional) demo / tests
 * ============================ */
static void demo2(uint32_t px, uint32_t py, int periodic_x, int periodic_y) {
  size_t total = (size_t)px * (size_t)py;
  Coord2 *coords = (Coord2 *)malloc(total * sizeof(Coord2));
  int *rankof = (int *)malloc(total * sizeof(int));
  build_zorder_coords2(px, py, coords);
  build_rank_of2(px, py, coords, rankof);

  printf("2D demo %ux%u (periodic_x=%d periodic_y=%d)\n", px, py, periodic_x,
         periodic_y);
  for (size_t r = 0; r < total; ++r) {
    uint32_t ry = coords[r].y, rx = coords[r].x;
    int left =
        z_neighbor_rank2(-1, 0, rx, ry, px, py, periodic_x, periodic_y, rankof);
    int right =
        z_neighbor_rank2(1, 0, rx, ry, px, py, periodic_x, periodic_y, rankof);
    int up =
        z_neighbor_rank2(0, 1, rx, ry, px, py, periodic_x, periodic_y, rankof);
    int down =
        z_neighbor_rank2(0, -1, rx, ry, px, py, periodic_x, periodic_y, rankof);
    printf("rank %2zu @ (y=%u,x=%u): L=%2d R=%2d U=%2d D=%2d\n", r, ry, rx,
           left, right, up, down);
  }
  free(coords);
  free(rankof);
}

static void demo3(uint32_t px, uint32_t py, uint32_t pz, int periodic_x,
                  int periodic_y, int periodic_z) {
  size_t total = (size_t)px * (size_t)py * (size_t)pz;
  Coord3 *coords = (Coord3 *)malloc(total * sizeof(Coord3));
  int *rankof = (int *)malloc(total * sizeof(int));
  build_zorder_coords3(px, py, pz, coords);
  build_rank_of3(px, py, pz, coords, rankof);

  printf("3D demo %ux%u×%u (periodic=%d,%d,%d)\n", px, py, pz, periodic_x,
         periodic_y, periodic_z);
  for (size_t r = 0; r < total; ++r) {
    uint32_t rz = coords[r].z, ry = coords[r].y, rx = coords[r].x;
    int xp = z_neighbor_rank3(1, 0, 0, rx, ry, rz, px, py, pz, periodic_x,
                              periodic_y, periodic_z, rankof);
    int xm = z_neighbor_rank3(-1, 0, 0, rx, ry, rz, px, py, pz, periodic_x,
                              periodic_y, periodic_z, rankof);
    int yp = z_neighbor_rank3(0, 1, 0, rx, ry, rz, px, py, pz, periodic_x,
                              periodic_y, periodic_z, rankof);
    int ym = z_neighbor_rank3(0, -1, 0, rx, ry, rz, px, py, pz, periodic_x,
                              periodic_y, periodic_z, rankof);
    int zp = z_neighbor_rank3(0, 0, 1, rx, ry, rz, px, py, pz, periodic_x,
                              periodic_y, periodic_z, rankof);
    int zm = z_neighbor_rank3(0, 0, -1, rx, ry, rz, px, py, pz, periodic_x,
                              periodic_y, periodic_z, rankof);
    printf(
        "rank %2zu @ (z=%u,y=%u,x=%u): x-=%2d x+=%2d y-=%2d y+=%2d z-=%2d "
        "z+=%2d\n",
        r, rz, ry, rx, xm, xp, ym, yp, zm, zp);
  }
  free(coords);
  free(rankof);
}

int main(void) {
  demo2(4, 3, 1, 0); /* 2D: px=4, py=3, periodic x, non-periodic y */
  puts("");
  demo3(2, 2, 2, 1, 1, 1); /* 3D: 2x2x2 all periodic */
  return 0;
}
