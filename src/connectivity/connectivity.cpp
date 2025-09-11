// C/C++
#include "connectivity.hpp"

#include <cstdlib>

void build_rank_of2(uint32_t px, uint32_t py, const Coord2 *coords,
                    int *rank_of_out) {
  const size_t total = (size_t)px * (size_t)py;
  for (size_t i = 0; i < total; ++i) rank_of_out[i] = -1;
  for (size_t r = 0; r < total; ++r) {
    const uint32_t y = coords[r].y;
    const uint32_t x = coords[r].x;
    rank_of_out[linear_index2(px, py, y, x)] = (int)r;
  }
}

void build_rank_of3(uint32_t px, uint32_t py, uint32_t pz, const Coord3 *coords,
                    int *rank_of_out) {
  const size_t total = (size_t)px * (size_t)py * (size_t)pz;
  for (size_t i = 0; i < total; ++i) rank_of_out[i] = -1;
  for (size_t r = 0; r < total; ++r) {
    const uint32_t z = coords[r].z;
    const uint32_t y = coords[r].y;
    const uint32_t x = coords[r].x;
    rank_of_out[linear_index3(px, py, z, y, x)] = (int)r;
  }
}

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

size_t build_zorder_coords3(uint32_t px, uint32_t py, uint32_t pz,
                            Coord3 *coords) {
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
