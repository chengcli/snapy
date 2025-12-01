// yaml
#include <yaml-cpp/yaml.h>

// snap
#include "forcing.hpp"

namespace snap {

RelaxBotCompOptions RelaxBotCompOptionsImpl::from_yaml(
    YAML::Node const& forcing) {
  if (!forcing["relax-bot-comp"]) return nullptr;

  auto node = forcing["relax-bot-comp"];
  auto op = RelaxBotCompOptionsImpl::create();

  op->tau() = node["tau"].as<double>(0.0);
  op->species() =
      node["species"].as<std::vector<std::string>>(std::vector<std::string>{});
  op->xfrac() = node["xfrac"].as<std::vector<double>>(std::vector<double>{});

  TORCH_CHECK(
      op->species().size() == op->xfrac().size(),
      "RelaxBotCompOptions: 'species' and 'xfrac' must have the same length.");

  return op;
}

torch::Tensor RelaxBotCompImpl::forward(torch::Tensor du, torch::Tensor w,
                                        torch::Tensor temp, double dt) {
  return du;
}

}  // namespace snap
