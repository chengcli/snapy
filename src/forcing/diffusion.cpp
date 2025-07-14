// yaml
#include <yaml-cpp/yaml.h>

// snap
#include <snap/snap.h>

#include "forcing.hpp"

namespace snap {

DiffusionOptions DiffusionOptions::from_yaml(YAML::Node const& node) {
  DiffusionOptions op;
  op.K() = node["K"].as<double>(0.);
  op.type() = node["type"].as<std::string>("theta");
  return op;
}

torch::Tensor DiffusionImpl::forward(torch::Tensor du, torch::Tensor w,
                                     torch::Tensor temp, double dt) {
  // Real temp = pthermo->GetTemp(w.at(pmb->ks, j, i));
  // Real theta = potential_temp(pthermo, w.at(pmb->ks, j, i), p0);

  return du;
}
}  // namespace snap
