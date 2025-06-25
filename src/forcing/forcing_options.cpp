// yaml
#include <yaml-cpp/yaml.h>

// snap
#include "forcing.hpp"

namespace snap {
ConstGravityOptions ConstGravityOptions::from_yaml(YAML::Node const& node) {
  ConstGravityOptions op;
  op.grav1() = node["grav1"].as<double>(0.);
  op.grav2() = node["grav2"].as<double>(0.);
  op.grav3() = node["grav3"].as<double>(0.);
  return op;
}

CoriolisOptions CoriolisOptions::from_yaml(YAML::Node const& node) {
  CoriolisOptions op;
  op.omega1() = node["omega1"].as<double>(0.);
  op.omega2() = node["omega2"].as<double>(0.);
  op.omega3() = node["omega3"].as<double>(0.);
  op.omegax() = node["omegax"].as<double>(0.);
  op.omegay() = node["omegay"].as<double>(0.);
  op.omegaz() = node["omegaz"].as<double>(0.);
  return op;
}

DiffusionOptions DiffusionOptions::from_yaml(YAML::Node const& node) {
  DiffusionOptions op;
  op.K() = node["K"].as<double>(0.);
  op.type() = node["type"].as<std::string>("theta");
  return op;
}
}  // namespace snap
