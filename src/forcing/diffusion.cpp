// yaml
#include <yaml-cpp/yaml.h>

// snap
#include "forcing.hpp"

namespace snap {

DiffusionOptions DiffusionOptionsImpl::from_yaml(YAML::Node const& forcing) {
  if (!forcing["diffusion"]) return nullptr;

  auto node = forcing["diffusion"];
  auto op = DiffusionOptionsImpl::create();

  op->K() = node["K"].as<double>(0.);
  op->type() = node["type"].as<std::string>("theta");

  return op;
}

torch::Tensor DiffusionImpl::forward(torch::Tensor du, torch::Tensor w,
                                     torch::Tensor temp, double dt) {
  // Real temp = pthermo->GetTemp(w.at(pmb->ks, j, i));
  // Real theta = potential_temp(pthermo, w.at(pmb->ks, j, i), p0);

  return du;
}
}  // namespace snap
