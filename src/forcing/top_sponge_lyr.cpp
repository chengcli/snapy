// yaml
#include <yaml-cpp/yaml.h>

// snap
#include "forcing.hpp"

namespace snap {

TopSpongeLyrOptions TopSpongeLyrOptions::from_yaml(YAML::Node const& node) {
  TopSpongeLyrOptions op;

  op.tau() = node["tau"].as<double>(0.0);
  op.width() = node["width"].as<double>(0.0);

  return op;
}

torch::Tensor TopSpongeLyrImpl::forward(torch::Tensor du, torch::Tensor w,
                                        torch::Tensor temp, double dt) {
  // Implement the top sponge layer logic here
  // For now, just return the input tensor
  return du;
}

}  // namespace snap
