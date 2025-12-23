// yaml
#include <yaml-cpp/yaml.h>

// snap
#include <snap/snap.h>

#include <snap/coord/coordinate.hpp>
#include <snap/hydro/hydro.hpp>

#include "forcing.hpp"

namespace snap {

BotSpongeLyrOptions BotSpongeLyrOptionsImpl::from_yaml(
    YAML::Node const& forcing) {
  if (!forcing["bot-sponge-lyr"]) return nullptr;

  auto node = forcing["bot-sponge-lyr"];
  auto op = BotSpongeLyrOptionsImpl::create();

  op->tau() = node["tau"].as<double>(0.0);
  op->width() = node["width"].as<double>(0.0);

  return op;
}

BotSpongeLyrImpl::BotSpongeLyrImpl(BotSpongeLyrOptions const& options_,
                                   torch::nn::Module* p)
    : options(options_) {
  phydro = dynamic_cast<HydroImpl const*>(p);
  reset();
}

void BotSpongeLyrImpl::reset() {
  TORCH_CHECK(phydro, "[BotSpongeLyr] Parent Hydro is null");
}

torch::Tensor BotSpongeLyrImpl::forward(torch::Tensor du, torch::Tensor w,
                                        torch::Tensor temp, double dt) {
  // Implement the bottom sponge layer logic here
  // For now, just return the input tensor
  return du;
}

}  // namespace snap
