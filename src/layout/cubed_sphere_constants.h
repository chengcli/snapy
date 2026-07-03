#pragma once

// base
#include <configure.h>

// snap
#include <snap/snap.h>

namespace snap {

enum { SIDE_L = 0, SIDE_R = 1, SIDE_B = 2, SIDE_T = 3 };

struct CSEdge {
  int nface; /* neighbor face id [0..5] */
  int nside; /* neighbor side id (LEFT/RIGHT/BOTTOM/TOP) */
  int rev;   /* 0: preserve along-edge index, 1: reverse */
};

struct CSVel {
  int idx; /* velocity component index */
  int sgn; /* velocity component sign: +1 or -1 */
};

inline DISPATCH_MACRO CSVel cs_cart_to_local_vel(int face, int component) {
  const int idx[6][3] = {
      {VEL3, VEL1, VEL2}, {VEL3, VEL2, VEL1}, {VEL3, VEL1, VEL2},
      {VEL1, VEL3, VEL2}, {VEL3, VEL2, VEL1}, {VEL1, VEL3, VEL2}};
  const int sgn[6][3] = {{+1, +1, +1}, {+1, -1, +1}, {+1, -1, -1},
                         {+1, -1, +1}, {+1, +1, -1}, {-1, +1, +1}};
  return {idx[face][component], sgn[face][component]};
}

inline DISPATCH_MACRO CSVel cs_local_to_cart_vel(int face, int component) {
  const int idx[6][3] = {
      {VEL2, VEL3, VEL1}, {VEL3, VEL2, VEL1}, {VEL2, VEL3, VEL1},
      {VEL1, VEL3, VEL2}, {VEL3, VEL2, VEL1}, {VEL1, VEL3, VEL2}};
  const int sgn[6][3] = {{+1, +1, +1}, {+1, -1, +1}, {-1, -1, +1},
                         {+1, +1, -1}, {-1, +1, +1}, {-1, +1, +1}};
  return {idx[face][component], sgn[face][component]};
}

template <typename T>
inline DISPATCH_MACRO void cs_face_xyz_from_tan(int face, T a, T b, T* x, T* y,
                                                T* z) {
  switch (face) {
    case 0:  // +X
      *x = T(1);
      *y = a;
      *z = b;
      break;
    case 1:  // +Y
      *x = -a;
      *y = T(1);
      *z = b;
      break;
    case 2:  // -X
      *x = T(-1);
      *y = -a;
      *z = b;
      break;
    case 3:  // +Z
      *x = -b;
      *y = a;
      *z = T(1);
      break;
    case 4:  // -Y
      *x = a;
      *y = T(-1);
      *z = b;
      break;
    default:  // -Z
      *x = b;
      *y = a;
      *z = T(-1);
      break;
  }
}

}  // namespace snap
