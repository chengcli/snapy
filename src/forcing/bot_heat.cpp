// yaml
#include <yaml-cpp/yaml.h>

// snap
#include <snap/snap.h>

#include <snap/registry.hpp>

#include "forcing.hpp"

namespace snap {

BotHeatOptions BotHeatOptions::from_yaml(YAML::Node const& node) {
  BotHeatOptions op;

  op.flux() = node["flux"].as<double>(0.0);

  return op;
}

void BotHeatImpl::reset() {
  pcoord = register_module_op(this, "coord", options.coord());
}

torch::Tensor BotHeatImpl::forward(torch::Tensor du, torch::Tensor w,
                                   torch::Tensor temp, double dt) {
  return du;
}

}  // namespace snap
