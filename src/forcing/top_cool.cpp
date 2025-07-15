// yaml
#include <yaml-cpp/yaml.h>

// snap
#include <snap/snap.h>

#include <snap/registry.hpp>

#include "forcing.hpp"

namespace snap {

TopCoolOptions TopCoolOptions::from_yaml(YAML::Node const& node) {
  TopCoolOptions op;

  op.flux() = node["flux"].as<double>(0.0);

  return op;
}

void TopCoolImpl::reset() {
  pcoord = register_module_op(this, "coord", options.coord());
}

torch::Tensor TopCoolImpl::forward(torch::Tensor du, torch::Tensor w,
                                   torch::Tensor temp, double dt) {
  return du;
}

}  // namespace snap
