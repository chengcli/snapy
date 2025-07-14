// yaml
#include <yaml-cpp/yaml.h>

// snap
#include "sedimentation.hpp"

namespace snap {

SedVelOptions SedVelOptions::from_yaml(YAML::Node const& root) {
  SedVelOptions op;

  if (!root["forcing"]) return op;
  if (!root["forcing"]["const-gravity"]) return op;

  op.grav() = root["forcing"]["const-gravity"]["grav1"].as<double>(0.0);

  auto config = root["sedimentation"];

  op.radius() = config["radius"].as<std::vector<double>>();
  op.density() = config["density"].as<std::vector<double>>();
  op.const_vsed() = config["const-vsed"].as<std::vector<double>>();

  op.a_diameter() = config["a-diameter"].as<double>(2.827e-10);
  op.a_epsilon_LJ() = config["a-epsilon-LJ"].as<double>(59.7e-7);
  op.a_mass() = config["a-mass"].as<double>(3.34e-27);
  op.min_radius() = config["min-radius"].as<double>(1.e-6);
  op.upper_limit() = config["upper-limit"].as<double>(5.e3);

  return op;
}

}  // namespace snap
