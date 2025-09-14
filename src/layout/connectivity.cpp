// snap
#include "connectivity.hpp"

namespace snap {

void build_rank_of2(int px, int py, const Coord2 *coords, int *rank_of_out) {
  const size_t total = (size_t)px * (size_t)py;
  for (size_t i = 0; i < total; ++i) rank_of_out[i] = -1;
  for (size_t r = 0; r < total; ++r) {
    const int y = coords[r].y;
    const int x = coords[r].x;
    rank_of_out[linear_index2(px, py, y, x)] = r;
  }
}

void build_rank_of3(int px, int py, int pz, const Coord3 *coords,
                    int *rank_of_out) {
  const size_t total = (size_t)px * (size_t)py * (size_t)pz;
  for (size_t i = 0; i < total; ++i) rank_of_out[i] = -1;
  for (size_t r = 0; r < total; ++r) {
    const int z = coords[r].z;
    const int y = coords[r].y;
    const int x = coords[r].x;
    rank_of_out[linear_index3(px, py, z, y, x)] = r;
  }
}

size_t build_zorder_coords2(int px, int py, Coord2 *coords) {
  const size_t need = (size_t)px * (size_t)py;
  size_t count = 0;
  int code = 0;
  while (count < need) {
    int y, x;
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

size_t build_zorder_coords3(int px, int py, int pz, Coord3 *coords) {
  const size_t need = (size_t)px * (size_t)py * (size_t)pz;
  size_t count = 0;
  uint64_t code = 0;
  while (count < need) {
    int z, y, x;
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

}  // namespace snap
