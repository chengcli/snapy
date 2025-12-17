// yaml
#include <yaml-cpp/yaml.h>

// snap
#include <snap/snap.h>

#include "forcing.hpp"

namespace snap {

BotHeatOptions BotHeatOptionsImpl::from_yaml(YAML::Node const& forcing) {
  if (!forcing["bot-heat"]) return nullptr;

  auto node = forcing["bot-heat"];
  auto op = BotHeatOptionsImpl::create();

  op->flux() = node["flux"].as<double>(0.0);
  op->depth() = node["depth"].as<int>(1);

  return op;
}

void BotHeatImpl::reset() {
  pcoord = CoordinateImpl::create(options->coord(), this);
}

torch::Tensor BotHeatImpl::forward(torch::Tensor du, torch::Tensor w,
                                   torch::Tensor temp, double dt) {
  int is = pcoord->is();
  auto dz = pcoord->dx1f[is];
  du[IPR].narrow(-1, is, options->depth()) +=
      options->flux() / (dz * options->depth());
  return du;
}

}  // namespace snap
