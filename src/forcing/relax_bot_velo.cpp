// yaml
#include <yaml-cpp/yaml.h>

// snap
#include "forcing.hpp"

namespace snap {

RelaxBotVeloOptions RelaxBotVeloOptions::from_yaml(YAML::Node const& node) {
  RelaxBotVeloOptions op;

  op.tau() = node["tau"].as<double>(0.0);
  op.bvx() = node["bvx"].as<double>(0.0);
  op.bvy() = node["bvy"].as<double>(0.0);
  op.bvz() = node["bvz"].as<double>(0.0);

  return op;
}

torch::Tensor RelaxBotVeloImpl::forward(torch::Tensor du, torch::Tensor w,
                                        torch::Tensor temp, double dt) {
  return du;
}

}  // namespace snap
