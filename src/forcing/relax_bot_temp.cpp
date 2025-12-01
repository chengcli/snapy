// yaml
#include <yaml-cpp/yaml.h>

// snap
#include "forcing.hpp"

namespace snap {

RelaxBotTempOptions RelaxBotTempOptionsImpl::from_yaml(
    YAML::Node const& forcing) {
  if (!forcing["relax-bot-temp"]) return nullptr;

  auto node = forcing["relax-bot-temp"];
  auto op = RelaxBotTempOptionsImpl::create();

  op->tau() = node["tau"].as<double>(0.0);
  op->btemp() = node["btemp"].as<double>(300.0);

  return op;
}

torch::Tensor RelaxBotTempImpl::forward(torch::Tensor du, torch::Tensor w,
                                        torch::Tensor temp, double dt) {
  return du;
}

}  // namespace snap
