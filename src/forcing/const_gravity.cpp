// yaml
#include <yaml-cpp/yaml.h>

// snap
#include <snap/snap.h>

#include <snap/input/check_keys.hpp>

#include "forcing.hpp"

namespace snap {
ConstGravityOptions ConstGravityOptionsImpl::from_yaml(
    YAML::Node const& forcing) {
  if (!forcing["const-gravity"]) return nullptr;

  auto node = forcing["const-gravity"];
  check_keys(node, "forcing/const-gravity",
             {"grav1", "grav2", "grav3", "non-hydrostatic"});
  auto op = ConstGravityOptionsImpl::create();

  op->grav1() = node["grav1"].as<double>(op->grav1());
  op->grav2() = node["grav2"].as<double>(op->grav2());
  op->grav3() = node["grav3"].as<double>(op->grav3());
  op->non_hydrostatic() =
      node["non-hydrostatic"].as<double>(op->non_hydrostatic());
  TORCH_CHECK(op->non_hydrostatic() >= 0. && op->non_hydrostatic() <= 1.);

  return op;
}

torch::Tensor ConstGravityImpl::forward(torch::Tensor du, torch::Tensor w,
                                        torch::Tensor temp, StepDt dt) {
  return forward_split(du, w, dt, w[IDN]);
}

torch::Tensor ConstGravityImpl::forward_split(torch::Tensor du, torch::Tensor w,
                                              StepDt dt, torch::Tensor rho1) {
  if (options->grav1() != 0.) {
    du[IVX] += dt * rho1 * options->grav1() * options->non_hydrostatic();
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
