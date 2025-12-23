// yaml
#include <yaml-cpp/yaml.h>

// snap
#include <snap/snap.h>

#include <snap/coord/coordinate.hpp>
#include <snap/hydro/hydro.hpp>

#include "forcing.hpp"

namespace snap {

TopSpongeLyrOptions TopSpongeLyrOptionsImpl::from_yaml(
    YAML::Node const& forcing) {
  if (!forcing["top-sponge-lyr"]) return nullptr;

  auto node = forcing["top-sponge-lyr"];
  auto op = TopSpongeLyrOptionsImpl::create();

  op->tau() = node["tau"].as<double>(0.0);
  op->width() = node["width"].as<double>(0.0);

  return op;
}

TopSpongeLyrImpl::TopSpongeLyrImpl(TopSpongeLyrOptions const& options_,
                                   torch::nn::Module* p)
    : options(options_) {
  phydro = dynamic_cast<HydroImpl const*>(p);
  reset();
}

void TopSpongeLyrImpl::reset() {
  TORCH_CHECK(phydro, "[TopSpongeLyr] Parent Hydro is null");
}

torch::Tensor TopSpongeLyrImpl::forward(torch::Tensor du, torch::Tensor w,
                                        torch::Tensor temp, double dt) {
  // Implement the top sponge layer logic here
  // For now, just return the input tensor
  return du;
}

}  // namespace snap
