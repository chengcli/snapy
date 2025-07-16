// snap
#include "cs_velocity_rotation.hpp"

namespace snap {

//! Transform cubed sphere velocity to local cartesian velocity
torch::Tensor vel_zab_to_zxy(torch::Tensor vel, torch::Tensor a,
                             torch::Tensor b) {
  torch::Tensor x = a.tan();
  torch::Tensor y = b.tan();

  torch::Tensor vx = vel[1];
  torch::Tensor vy = vel[2];
  torch::Tensor vz = vel[0];

  torch::Tensor delta = sqrt(x * x + y * y + 1);
  torch::Tensor C = sqrt(1 + x * x);
  torch::Tensor D = sqrt(1 + y * y);

  auto result = torch::empty_like(vel);

  result[0] = (vz - D * x * vx - C * y * vy) / delta;
  result[1] = (x * vz + D * vx) / delta;
  result[2] = (y * vz + C * vy) / delta;

  return result;
}

//! Transform local cartesian velocity to cubed sphere velocity
torch::Tensor vel_zxy_to_zab(torch::Tensor vel, torch::Tensor a,
                             torch::Tensor b) {
  torch::Tensor x = a.tan();
  torch::Tensor y = b.tan();

  torch::Tensor vx = vel[1];
  torch::Tensor vy = vel[2];
  torch::Tensor vz = vel[0];

  torch::Tensor delta = sqrt(x * x + y * y + 1);
  torch::Tensor C = sqrt(1 + x * x);
  torch::Tensor D = sqrt(1 + y * y);

  auto result = torch::empty_like(vel);

  result[0] = (vz + x * vx + y * vy) / delta;
  result[1] = (-x * vz / D + vx * (1 + y * y) / D - vy * x * y / D) / delta;
  result[2] = (-y * vz / C - x * y * vx / C + (1 + x * x) * vy / C) / delta;
  return result;
}

//! Transform cubed sphere velocity from panel 1 to panel 2
//! \param a $x = \tan(\xi)$ coordinates
//! \param b $y = \tan(\eta)$ coordinat
torch::Tensor vel_zab_from_p1(torch::Tensor vel, torch::Tensor a,
                              torch::Tensor b, int panel) {
  auto result = vel_zab_to_zxy(vel, a, b);
  torch::Tensor v1 = result[0].clone();
  torch::Tensor v2 = result[1].clone();
  torch::Tensor v3 = result[2].clone();

  switch (panel) {
    case 2:
      // z->y, x->-x, y->z
      result[0] = v3;
      result[1] = -v2;
      result[2] = v1;
      return vel_zxy_to_zab(result, a, b);
    case 3:
      // z->-x, x->z, y->y
      result[0] = v2;
      result[1] = -v1;
      result[2] = v3;
      return vel_zxy_to_zab(result, a, b);
    case 4:
      // z->-x, x->-y, y->z
      result[0] = -v2;
      result[1] = -v3;
      result[2] = v1;
      return vel_zxy_to_zab(result, a, b);
    case 6:
      // z->-y, x->x, y->z
      result[0] = -v3;
      result[1] = v2;
      result[2] = v1;
      return vel_zxy_to_zab(result, a, b);
    default:
      TORCH_CHECK(false, "Unallowed transformation");
  }
}

torch::Tensor vel_zab_from_p2(torch::Tensor vel, torch::Tensor a,
                              torch::Tensor b, int panel) {
  auto result = vel_zab_to_zxy(vel, a, b);
  torch::Tensor v1 = result[0].clone();
  torch::Tensor v2 = result[1].clone();
  torch::Tensor v3 = result[2].clone();
  switch (panel) {
    case 1:
      // z->y, x->-x, y->z
      result[0] = v3;
      result[1] = -v2;
      result[2] = v1;
      return vel_zxy_to_zab(result, a, b);
    case 3:
      // z->-x, x->-y, y->z
      result[0] = -v2;
      result[1] = -v3;
      result[2] = v1;
      return vel_zxy_to_zab(result, a, b);
    case 4:
      // z->x, x->-z, y->y
      result[0] = v2;
      result[1] = -v1;
      result[2] = v3;
      return vel_zxy_to_zab(result, a, b);
    case 5:
      // z->-y, x->x, y->z
      result[0] = -v3;
      result[1] = v2;
      result[2] = v1;
      return vel_zxy_to_zab(result, a, b);
    default:
      TORCH_CHECK(false, "Unallowed transformation");
  }
}

torch::Tensor vel_zab_from_p3(torch::Tensor vel, torch::Tensor a,
                              torch::Tensor b, int panel) {
  auto result = vel_zab_to_zxy(vel, a, b);
  torch::Tensor v1 = result[0].clone();
  torch::Tensor v2 = result[1].clone();
  torch::Tensor v3 = result[2].clone();
  switch (panel) {
    case 1:
      // z->-x, x->z, y->y
      result[0] = -v2;
      result[1] = v1;
      result[2] = v3;
      return vel_zxy_to_zab(result, a, b);
    case 2:
      // z->y, x->-z, y->-x
      result[0] = v3;
      result[1] = -v1;
      result[2] = -v2;
      return vel_zxy_to_zab(result, a, b);
    case 5:
      // z->x, x->-z, y->y
      result[0] = v2;
      result[1] = -v1;
      result[2] = v3;
      return vel_zxy_to_zab(result, a, b);
    case 6:
      // z->-y, x->z, y->-x
      result[0] = -v3;
      result[1] = v1;
      result[2] = -v2;
      return vel_zxy_to_zab(result, a, b);
    default:
      TORCH_CHECK(false, "Unallowed transformation");
  }
}

torch::Tensor vel_zab_from_p4(torch::Tensor vel, torch::Tensor a,
                              torch::Tensor b, int panel) {
  auto result = vel_zab_to_zxy(vel, a, b);
  torch::Tensor v1 = result[0].clone();
  torch::Tensor v2 = result[1].clone();
  torch::Tensor v3 = result[2].clone();
  switch (panel) {
    case 1:
      // z->y, x->-z, y->-x
      result[0] = v3;
      result[1] = -v1;
      result[2] = -v2;
      return vel_zxy_to_zab(result, a, b);
    case 2:
      // z->-x, x->z, y->y
      result[0] = -v2;
      result[1] = v1;
      result[2] = v3;
      return vel_zxy_to_zab(result, a, b);
    case 5:
      // z->-y, x->z, y->-x
      result[0] = -v3;
      result[1] = v1;
      result[2] = -v2;
      return vel_zxy_to_zab(result, a, b);
    case 6:
      // z->x, x->-z, y->y
      result[0] = v2;
      result[1] = -v1;
      result[2] = v3;
      return vel_zxy_to_zab(result, a, b);
    default:
      TORCH_CHECK(false, "Unallowed transformation");
  }
}

torch::Tensor vel_zab_from_p5(torch::Tensor vel, torch::Tensor a,
                              torch::Tensor b, int panel) {
  auto result = vel_zab_to_zxy(vel, a, b);
  torch::Tensor v1 = result[0].clone();
  torch::Tensor v2 = result[1].clone();
  torch::Tensor v3 = result[2].clone();
  switch (panel) {
    case 2:
      // z->y, x->x, y->-z
      result[0] = v3;
      result[1] = v2;
      result[2] = -v1;
      return vel_zxy_to_zab(result, a, b);
    case 3:
      // z->-x, x->z, y->y
      result[0] = -v2;
      result[1] = v1;
      result[2] = v3;
      return vel_zxy_to_zab(result, a, b);
    case 4:
      // z->x, x->-y, y->-z
      result[0] = v2;
      result[1] = -v3;
      result[2] = -v1;
      return vel_zxy_to_zab(result, a, b);
    case 6:
      // z->-y, x->-x, y->-z
      result[0] = -v3;
      result[1] = -v2;
      result[2] = -v1;
      return vel_zxy_to_zab(result, a, b);
    default:
      TORCH_CHECK(false, "Unallowed transformation");
  }
}

torch::Tensor vel_zab_from_p6(torch::Tensor vel, torch::Tensor a,
                              torch::Tensor b, int panel) {
  auto result = vel_zab_to_zxy(vel, a, b);
  torch::Tensor v1 = result[0].clone();
  torch::Tensor v2 = result[1].clone();
  torch::Tensor v3 = result[2].clone();
  switch (panel) {
    case 1:
      // z->y, x->x, y->-z
      result[0] = v3;
      result[1] = v2;
      result[2] = -v1;
      return vel_zxy_to_zab(result, a, b);
    case 3:
      // z->x, x->-y, y->-z
      result[0] = v2;
      result[1] = -v3;
      result[2] = -v1;
      return vel_zxy_to_zab(result, a, b);
    case 4:
      // z->-x, x->z, y->y
      result[0] = -v2;
      result[1] = v1;
      result[2] = v3;
      return vel_zxy_to_zab(result, a, b);
    case 5:
      // z->-y, x->-x, y->-z
      result[0] = -v3;
      result[1] = -v2;
      result[2] = -v1;
      return vel_zxy_to_zab(result, a, b);
    default:
      TORCH_CHECK(false, "Unallowed transformation");
  }
}
}  // namespace snap
