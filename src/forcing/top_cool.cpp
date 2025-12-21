// yaml
#include <yaml-cpp/yaml.h>

// snap
#include <snap/snap.h>

#include "forcing.hpp"

namespace snap {

TopCoolOptions TopCoolOptionsImpl::from_yaml(YAML::Node const& forcing) {
  if (!forcing["top-cool"]) return nullptr;

  auto node = forcing["top-cool"];
  auto op = TopCoolOptionsImpl::create();

  op->flux() = node["flux"].as<double>(0.0);
  op->depth() = node["depth"].as<int>(1);

  TORCH_CHECK(op->flux() <= 0., "TopCool flux must be negative");
  TORCH_CHECK(op->depth() > 0., "TopCool depth must be greater than zero");

  return op;
}

void TopCoolImpl::reset() {
  pcoord = CoordinateImpl::create(options->coord(), this);
}

torch::Tensor TopCoolImpl::forward(torch::Tensor du, torch::Tensor w,
                                   torch::Tensor temp, double dt) {
  int iu = pcoord->iu();
  auto dz = pcoord->dx1f[iu];
  du[IPR].slice(-1, iu + 1 - options->depth(), iu + 1) +=
      options->flux() / (dz * options->depth());
  return du;
}

}  // namespace snap
