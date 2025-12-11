// torch
#include <torch/torch.h>

// snap
#include <snap/snap.h>

namespace snap {

void sph_cart_to_contra_(torch::Tensor const& vel, torch::Tensor theta,
                         torch::Tensor phi) {
  auto vz = vel[VEL1].clone();
  auto vx = vel[VEL2].clone();
  auto vy = vel[VEL3].clone();

  vel[VEL1] = vx * theta.sin() * phi.cos() + vy * theta.sin() * phi.sin() +
              vz * theta.cos();
  vel[VEL2] = vx * theta.cos() * phi.cos() + vy * theta.cos() * phi.sin() -
              vz * theta.sin();
  vel[VEL3] = -vx * phi.sin() + vy * phi.cos();
}

void sph_contra_to_cart_(torch::Tensor const& vel, torch::Tensor theta,
                         torch::Tensor phi) {
  auto vr = vel[VEL1].clone();
  auto vt = vel[VEL2].clone();
  auto vp = vel[VEL3].clone();

  vel[VEL2] = vr * theta.sin() * phi.cos() + vt * theta.cos() * phi.cos() -
              vp * phi.sin();
  vel[VEL3] = vr * theta.sin() * phi.sin() + vt * theta.cos() * phi.sin() +
              vp * phi.cos();
  vel[VEL1] = vr * theta.cos() - vt * theta.sin();
}

}  // namespace snap
