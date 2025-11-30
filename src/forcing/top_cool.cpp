// yaml
#include <yaml-cpp/yaml.h>

// snap
#include <snap/snap.h>

#include "forcing.hpp"

namespace snap {

TopCoolOptions TopCoolOptionsImpl::from_yaml(YAML::Node const& node) {
  auto op = TopCoolOptionsImpl::create();

  op->flux() = node["flux"].as<double>(0.0);

  return op;
}

void TopCoolImpl::reset() {
  CHECK_MODULE_LINKED(TopCoolOptions, coord);
  pcoord = CoordinateImpl::create(options->coord(), this);
}

torch::Tensor TopCoolImpl::forward(torch::Tensor du, torch::Tensor w,
                                   torch::Tensor temp, double dt) {
  return du;
}

}  // namespace snap
