// torch
#include <torch/torch.h>

// snap
#include <snap/snap.h>

namespace snap {

void sph_cart_to_contra_(torch::Tensor const& vel, torch::Tensor theta,
                         torch::Tensor phi) {
  auto vz = vel[VEL_Z].clone();
  auto vx = vel[VEL_X].clone();
  auto vy = vel[VEL_Y].clone();

  vel[VEL_R] = vx * theta.sin() * phi.cos() + vy * theta.sin() * phi.sin() +
               vz * theta.cos();
  vel[VEL_T] = vx * theta.cos() * phi.cos() + vy * theta.cos() * phi.sin() -
               vz * theta.sin();
  vel[VEL_P] = -vx * phi.sin() + vy * phi.cos();
}

void sph_contra_to_cart_(torch::Tensor const& vel, torch::Tensor theta,
                         torch::Tensor phi) {
  auto vr = vel[VEL_R].clone();
  auto vt = vel[VEL_T].clone();
  auto vp = vel[VEL_P].clone();

  vel[VEL_Z] = vr * theta.sin() * phi.cos() + vt * theta.cos() * phi.cos() -
               vp * phi.sin();
  vel[VEL_X] = vr * theta.sin() * phi.sin() + vt * theta.cos() * phi.sin() +
               vp * phi.cos();
  vel[VEL_Y] = vr * theta.cos() - vt * theta.sin();
}

}  // namespace snap
