// yaml
#include <yaml-cpp/yaml.h>

// snap
#include <snap/snap.h>

#include "forcing.hpp"

namespace snap {
ConstGravityOptions ConstGravityOptionsImpl::from_yaml(
    YAML::Node const& forcing) {
  if (!forcing["const-gravity"]) return nullptr;

  auto node = forcing["const-gravity"];
  auto op = ConstGravityOptionsImpl::create();

  op->grav1() = node["grav1"].as<double>(0.);
  op->grav2() = node["grav2"].as<double>(0.);
  op->grav3() = node["grav3"].as<double>(0.);
  op->non_hydrostatic() = node["non-hydrostatic"].as<double>(1.);
  TORCH_CHECK(op->non_hydrostatic() >= 0. && op->non_hydrostatic() <= 1.);

  return op;
}

torch::Tensor ConstGravityImpl::forward(torch::Tensor du, torch::Tensor w,
                                        torch::Tensor temp, double dt) {
  if (options->grav1() != 0.) {
    du[IVX] += dt * w[IDN] * options->grav1() * options->non_hydrostatic();
    du[IPR] +=
        dt * w[IDN] * w[IVX] * options->grav1() * options->non_hydrostatic();
  }

  if (options->grav2() != 0.) {
    du[IVY] += dt * w[IDN] * options->grav2();
    du[IPR] += dt * w[IDN] * w[IVY] * options->grav2();
  }

  if (options->grav3() != 0.) {
    du[IVZ] += dt * w[IDN] * options->grav3();
    du[IPR] += dt * w[IDN] * w[IVZ] * options->grav3();
  }

  return du;
}
}  // namespace snap
