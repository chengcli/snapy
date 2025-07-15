// yaml
#include <yaml-cpp/yaml.h>

// snap
#include "forcing.hpp"

namespace snap {

RelaxBotTempOptions RelaxBotTempOptions::from_yaml(YAML::Node const& node) {
  RelaxBotTempOptions op;

  op.tau() = node["tau"].as<double>(0.0);
  op.btemp() = node["btemp"].as<double>(300.0);

  return op;
}

torch::Tensor RelaxBotTempImpl::forward(torch::Tensor du, torch::Tensor w,
                                        torch::Tensor temp, double dt) {
  return du;
}

}  // namespace snap
