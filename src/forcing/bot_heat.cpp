// yaml
#include <yaml-cpp/yaml.h>

// snap
#include <snap/snap.h>

#include "forcing.hpp"

namespace snap {

BotHeatOptions BotHeatOptionsImpl::from_yaml(YAML::Node const& node) {
  auto op = BotHeatOptionsImpl::create();

  op->flux() = node["flux"].as<double>(0.0);

  return op;
}

void BotHeatImpl::reset() {
  CHECK_MODULE_LINKED(BotHeatOptions, coord);
  pcoord = CoordinateImpl::create(options->coord(), this);
}

torch::Tensor BotHeatImpl::forward(torch::Tensor du, torch::Tensor w,
                                   torch::Tensor temp, double dt) {
  return du;
}

}  // namespace snap
