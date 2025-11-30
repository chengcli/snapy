// yaml
#include <yaml-cpp/yaml.h>

// snap
#include "forcing.hpp"

namespace snap {

PlumeForcingOptions PlumeForcingOptionsImpl::from_yaml(
    YAML::Node const& forcing) {
  if (!forcing["plume-forcing"]) return nullptr;

  auto node = forcing["plume-forcing"];
  auto op = PlumeForcingOptionsImpl::create();

  op->entrainment() = node["entrainment"].as<double>(0.1);
  op->N2() = node["N2"].as<double>(0.0);

  return op;
}

//! Plume forcing implementation
/*!
 * Primitive variables: w = [R, W, B, V]
 */
torch::Tensor PlumeForcingImpl::forward(torch::Tensor du, torch::Tensor w,
                                        torch::Tensor temp, double dt) {
  auto R = w[0];
  auto W = w[1];
  auto B = w[2];
  auto V = w[3];

  du[0] -= 2. * R * options->entrainment() * dt;
  du[1] += R * R * B * dt;
  du[2] -= options->N2() * R * R * W * dt;

  return du;
}

}  // namespace snap
