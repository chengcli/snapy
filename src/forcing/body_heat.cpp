// yaml
#include <yaml-cpp/yaml.h>

// snap
#include "forcing.hpp"

namespace snap {

BodyHeatOptions BodyHeatOptions::from_yaml(YAML::Node const& node) {
  BodyHeatOptions op;

  op.dTdt() = node["dTdt"].as<double>(0.0);
  op.pmin() = node["pmin"].as<double>(0.0);
  op.pmax() = node["pmax"].as<double>(1.0e6);

  return op;
}

void BodyHeatImpl::reset() {
  pthermo = register_module("thermo", kintera::ThermoY(options.thermo()));
}

torch::Tensor BodyHeatImpl::forward(torch::Tensor du, torch::Tensor w,
                                    torch::Tensor temp, double dt) {
  return du;
}

}  // namespace snap
