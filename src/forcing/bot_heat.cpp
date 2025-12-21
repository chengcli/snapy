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

  TORCH_CHECK(op->flux() >= 0., "BotHeat flux must be positive");
  TORCH_CHECK(op->depth() > 0., "BotHeat depth must be greater than zero");

  return op;
}

void BotHeatImpl::reset() {
  pcoord = CoordinateImpl::create(options->coord(), this);
}

torch::Tensor BotHeatImpl::forward(torch::Tensor du, torch::Tensor w,
                                   torch::Tensor temp, double dt) {
  int il = pcoord->il();
  auto dz = pcoord->dx1f[il];
  du[IPR].narrow(-1, il, options->depth()) +=
      options->flux() / (dz * options->depth());
  return du;
}

}  // namespace snap
