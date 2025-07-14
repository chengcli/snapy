// yaml
#include <yaml-cpp/yaml.h>

// snap
#include <snap/snap.h>

#include "forcing.hpp"

namespace snap {
ConstGravityOptions ConstGravityOptions::from_yaml(YAML::Node const& node) {
  ConstGravityOptions op;
  op.grav1() = node["grav1"].as<double>(0.);
  op.grav2() = node["grav2"].as<double>(0.);
  op.grav3() = node["grav3"].as<double>(0.);
  return op;
}

torch::Tensor ConstGravityImpl::forward(torch::Tensor du, torch::Tensor w,
                                        torch::Tensor temp, double dt) {
  if (options.grav1() != 0.) {
    du[Index::IVX] += dt * w[Index::IDN] * options.grav1();
    du[Index::IPR] += dt * w[Index::IDN] * w[Index::IVX] * options.grav1();
  }

  if (options.grav2() != 0.) {
    du[Index::IVY] += dt * w[Index::IDN] * options.grav2();
    du[Index::IPR] += dt * w[Index::IDN] * w[Index::IVY] * w[Index::IVY];
  }

  if (options.grav3() != 0.) {
    du[Index::IVZ] += dt * w[Index::IDN] * options.grav3();
    du[Index::IPR] += dt * w[Index::IDN] * w[Index::IVZ] * w[Index::IVZ];
  }

  return du;
}
}  // namespace snap
