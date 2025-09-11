/* zorder_neighbors.c
 *
 * Z-order (Morton) helpers and neighbor rank queries for 2D/3D process grids.
 * - Works for arbitrary grid sizes (need not be powers of two).
 * - Periodic/non-periodic handled independently per dimension.
 * - Off-domain neighbors on non-periodic edges return -1.
 *
 * Compile:  cc -O3 -std=c99 zorder_neighbors.c -o zorder_neighbors
 */

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ===========================
 * Bit de-interleaving helpers
 * ===========================
 * compact1by1:  remove every other bit (keep bits at positions 0,2,4,...)
 * compact1by2:  remove two other bits (keep bits at positions 0,3,6,...) for 3-way interleave
 *
 * Based on "Bit Twiddling Hacks" (public domain).
 */

static inline uint32_t compact1by1(uint32_t v) {
    v &= 0x55555555u;
    v = (v | (v >> 1)) & 0x33333333u;
    v = (v | (v >> 2)) & 0x0F0F0F0Fu;
    v = (v | (v >> 4)) & 0x00FF00FFu;
    v = (v | (v >> 8)) & 0x0000FFFFu;
    return v;
}

/* For 3D Morton codes we need to extract every 3rd bit. Using 64-bit lanes. */
static inline uint32_t compact1by2(uint64_t v) {
    v &= 0x1249249249249249ULL;                      // 0b001001.. pattern
    v = (v ^ (v >> 2)) & 0x10c30c30c30c30c3ULL;
    v = (v ^ (v >> 4)) & 0x100f00f00f00f00fULL;
    v = (v ^ (v >> 8)) & 0x1f0000ff0000ffULL;
    v = (v ^ (v >> 16)) & 0x1f00000000ffffULL;
    v = (v ^ (v >> 32)) & 0x1fffffULL;
    return (uint32_t)v;
}

/* ===========================
 * Morton decode (Z-order)
 * =========================== */

/* Decode 2D Morton code -> (y,x). Note the order: (y,x). */
static inline void morton_decode2(uint32_t code, uint32_t *y, uint32_t *x) {
    *x = compact1by1(code);
    *y = compact1by1(code >> 1);
}

/* Decode 3D Morton code -> (z,y,x). Uses 64-bit Morton codes. */
static inline void morton_decode3(uint64_t code, uint32_t *z, uint32_t *y, uint32_t *x) {
    *x = compact1by2(code);
    *y = compact1by2(code >> 1);
    *z = compact1by2(code >> 2);
}

/* ===========================
 * Coordinate containers
 * =========================== */

typedef struct { uint32_t y, x; } Coord2;
typedef struct { uint32_t z, y, x; } Coord3;

/* ================
 * Z-order builders
 * ================ */

/* Build Py×Px coordinates in Z-order. coords must have length >= py*px. */
size_t build_zorder_coords2(uint32_t px, uint32_t py, Coord2 *coords) {
    const size_t need = (size_t)px * (size_t)py;
    size_t count = 0;
    uint32_t code = 0;
    while (count < need) {
        uint32_t y, x;
        morton_decode2(code, &y, &x);
        if (x < px && y < py) {
            coords[count].y = y;
            coords[count].x = x;
            ++count;
        }
        ++code;
    }
    return count; /* == need */
}

/* Build Pz×Py×Px coordinates in Z-order. coords must have length >= pz*py*px. */
size_t build_zorder_coords3(uint32_t px, uint32_t py, uint32_t pz, Coord3 *coords) {
    const size_t need = (size_t)px * (size_t)py * (size_t)pz;
    size_t count = 0;
    uint64_t code = 0;
    while (count < need) {
        uint32_t z, y, x;
        morton_decode3(code, &z, &y, &x);
        if (x < px && y < py && z < pz) {
            coords[count].z = z;
            coords[count].y = y;
            coords[count].x = x;
            ++count;
        }
        ++code;
    }
    return count; /* == need */
}

/* ======================
 * coords -> rank mapping
 * ======================
 * We build a dense array rank_of with shape:
 *  - 2D: [py][px]
 *  - 3D: [pz][py][px]
 * storing the Z-order rank at that coordinate.
 * Access via linear index helpers below.
 */

static inline size_t index2(uint32_t px, uint32_t /*py*/, uint32_t y, uint32_t x) {
    return (size_t)y * (size_t)px + (size_t)x;
}
static inline size_t index3(uint32_t px, uint32_t py, uint32_t z, uint32_t y, uint32_t x) {
    return ((size_t)z * (size_t)py + (size_t)y) * (size_t)px + (size_t)x;
}

/* rank_of2: array length py*px, filled with rank at (y,x) */
void build_rank_of2(uint32_t px, uint32_t py, const Coord2 *coords, int *rank_of_out) {
    const size_t total = (size_t)px * (size_t)py;
    for (size_t i = 0; i < total; ++i) rank_of_out[i] = -1;
    for (size_t r = 0; r < total; ++r) {
        const uint32_t y = coords[r].y;
        const uint32_t x = coords[r].x;
        rank_of_out[index2(px, py, y, x)] = (int)r;
    }
}

/* rank_of3: array length pz*py*px, filled with rank at (z,y,x) */
void build_rank_of3(uint32_t px, uint32_t py, uint32_t pz, const Coord3 *coords, int *rank_of_out) {
    const size_t total = (size_t)px * (size_t)py * (size_t)pz;
    for (size_t i = 0; i < total; ++i) rank_of_out[i] = -1;
    for (size_t r = 0; r < total; ++r) {
        const uint32_t z = coords[r].z;
        const uint32_t y = coords[r].y;
        const uint32_t x = coords[r].x;
        rank_of_out[index3(px, py, z, y, x)] = (int)r;
    }
}

/* ============================
 * Neighbor → Z-order rank (2D)
 * ============================
 * dx,dy ∈ {-1,0,1}. periodic flags control wrap; otherwise off-domain → -1.
 * (rx,ry) are THIS rank's coords in the process grid (not Morton code).
 */

int z_neighbor_rank2(
    int dx, int dy,
    uint32_t rx, uint32_t ry,
    uint32_t px, uint32_t py,
    int periodic_x, int periodic_y,
    const int *rank_of /* length py*px */
) {
    int nx = (int)rx + dx;
    int ny = (int)ry + dy;

    if (periodic_x) {
        if      (nx < 0)       nx += (int)px;
        else if (nx >= (int)px) nx -= (int)px;
    } else {
        if (nx < 0 || nx >= (int)px) return -1;
    }

    if (periodic_y) {
        if      (ny < 0)       ny += (int)py;
        else if (ny >= (int)py) ny -= (int)py;
    } else {
        if (ny < 0 || ny >= (int)py) return -1;
    }

    return rank_of[index2(px, py, (uint32_t)ny, (uint32_t)nx)];
}

/* ============================
 * Neighbor → Z-order rank (3D)
 * ============================
 * dx,dy,dz ∈ {-1,0,1}. periodic flags control wrap; otherwise off-domain → -1.
 * (rx,ry,rz) are THIS rank's coords in the process grid (not Morton code).
 */

int z_neighbor_rank3(
    int dx, int dy, int dz,
    uint32_t rx, uint32_t ry, uint32_t rz,
    uint32_t px, uint32_t py, uint32_t pz,
    int periodic_x, int periodic_y, int periodic_z,
    const int *rank_of /* length pz*py*px */
) {
    int nx = (int)rx + dx;
    int ny = (int)ry + dy;
    int nz = (int)rz + dz;

    if (periodic_x) {
        if      (nx < 0)       nx += (int)px;
        else if (nx >= (int)px) nx -= (int)px;
    } else {
        if (nx < 0 || nx >= (int)px) return -1;
    }

    if (periodic_y) {
        if      (ny < 0)       ny += (int)py;
        else if (ny >= (int)py) ny -= (int)py;
    } else {
        if (ny < 0 || ny >= (int)py) return -1;
    }

    if (periodic_z) {
        if      (nz < 0)       nz += (int)pz;
        else if (nz >= (int)pz) nz -= (int)pz;
    } else {
        if (nz < 0 || nz >= (int)pz) return -1;
    }

    return rank_of[index3(px, py, (uint32_t)nz, (uint32_t)ny, (uint32_t)nx)];
}

/* ============================
 * (Optional) demo / tests
 * ============================ */
static void demo2(uint32_t px, uint32_t py, int periodic_x, int periodic_y) {
    size_t total = (size_t)px * (size_t)py;
    Coord2 *coords = (Coord2*)malloc(total * sizeof(Coord2));
    int    *rankof = (int*)   malloc(total * sizeof(int));
    build_zorder_coords2(px, py, coords);
    build_rank_of2(px, py, coords, rankof);

    printf("2D demo %ux%u (periodic_x=%d periodic_y=%d)\n", px, py, periodic_x, periodic_y);
    for (size_t r = 0; r < total; ++r) {
        uint32_t ry = coords[r].y, rx = coords[r].x;
        int left  = z_neighbor_rank2(-1,  0, rx, ry, px, py, periodic_x, periodic_y, rankof);
        int right = z_neighbor_rank2( 1,  0, rx, ry, px, py, periodic_x, periodic_y, rankof);
        int up    = z_neighbor_rank2( 0,  1, rx, ry, px, py, periodic_x, periodic_y, rankof);
        int down  = z_neighbor_rank2( 0, -1, rx, ry, px, py, periodic_x, periodic_y, rankof);
        printf("rank %2zu @ (y=%u,x=%u): L=%2d R=%2d U=%2d D=%2d\n", r, ry, rx, left, right, up, down);
    }
    free(coords); free(rankof);
}

static void demo3(uint32_t px, uint32_t py, uint32_t pz, int periodic_x, int periodic_y, int periodic_z) {
    size_t total = (size_t)px * (size_t)py * (size_t)pz;
    Coord3 *coords = (Coord3*)malloc(total * sizeof(Coord3));
    int    *rankof = (int*)   malloc(total * sizeof(int));
    build_zorder_coords3(px, py, pz, coords);
    build_rank_of3(px, py, pz, coords, rankof);

    printf("3D demo %ux%u×%u (periodic=%d,%d,%d)\n", px, py, pz, periodic_x, periodic_y, periodic_z);
    for (size_t r = 0; r < total; ++r) {
        uint32_t rz = coords[r].z, ry = coords[r].y, rx = coords[r].x;
        int xp = z_neighbor_rank3( 1, 0, 0, rx, ry, rz, px, py, pz, periodic_x, periodic_y, periodic_z, rankof);
        int xm = z_neighbor_rank3(-1, 0, 0, rx, ry, rz, px, py, pz, periodic_x, periodic_y, periodic_z, rankof);
        int yp = z_neighbor_rank3( 0, 1, 0, rx, ry, rz, px, py, pz, periodic_x, periodic_y, periodic_z, rankof);
        int ym = z_neighbor_rank3( 0,-1, 0, rx, ry, rz, px, py, pz, periodic_x, periodic_y, periodic_z, rankof);
        int zp = z_neighbor_rank3( 0, 0, 1, rx, ry, rz, px, py, pz, periodic_x, periodic_y, periodic_z, rankof);
        int zm = z_neighbor_rank3( 0, 0,-1, rx, ry, rz, px, py, pz, periodic_x, periodic_y, periodic_z, rankof);
        printf("rank %2zu @ (z=%u,y=%u,x=%u): x-=%2d x+=%2d y-=%2d y+=%2d z-=%2d z+=%2d\n",
               r, rz, ry, rx, xm, xp, ym, yp, zm, zp);
    }
    free(coords); free(rankof);
}

int main(void) {
    demo2(4, 3, 1, 0);  /* 2D: px=4, py=3, periodic x, non-periodic y */
    puts("");
    demo3(2, 2, 2, 1, 1, 1); /* 3D: 2x2x2 all periodic */
    return 0;
}
