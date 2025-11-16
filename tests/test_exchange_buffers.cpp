// C/C++
#include <cstdio>

// snap
#include <snap/layout/exchange.hpp>

int main(void) {
  printf("Testing get_buffer_id function\n");
  printf("================================\n\n");

  // Test 2D layout (dz=0)
  printf("2D layout (9 buffers):\n");
  for (int dy = -1; dy <= 1; ++dy) {
    for (int dx = -1; dx <= 1; ++dx) {
      int bid = snap::get_buffer_id(dx, dy, 0);
      printf("  offset (%2d,%2d, 0) -> buffer %d\n", dx, dy, bid);
    }
  }
  printf("\n");

  // Test 3D layout (full 27 neighbors)
  printf("3D layout (27 buffers):\n");
  for (int dz = -1; dz <= 1; ++dz) {
    printf("  z-offset = %d:\n", dz);
    for (int dy = -1; dy <= 1; ++dy) {
      for (int dx = -1; dx <= 1; ++dx) {
        int bid = snap::get_buffer_id(dx, dy, dz);
        printf("    offset (%2d,%2d,%2d) -> buffer %2d\n", dx, dy, dz, bid);
      }
    }
  }

  // Verify center is always buffer 0 (for all cases where all offsets are 0)
  int center = snap::get_buffer_id(0, 0, 0);
  printf("\nCenter buffer: %d (expected 0)\n", center);
  
  // Verify corners in 2D
  int corner_2d_tl = snap::get_buffer_id(-1, 1, 0);
  int corner_2d_br = snap::get_buffer_id(1, -1, 0);
  printf("\nTop-left corner (2D): %d (expected 5)\n", corner_2d_tl);
  printf("Bottom-right corner (2D): %d (expected 7)\n", corner_2d_br);

  return 0;
}
