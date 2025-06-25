// yaml
#include <yaml-cpp/yaml.h>

// snap
#include "scalar.hpp"

namespace snap {

ScalarOptions ScalarOptions::from_yaml(std::string const& filename) {
  ScalarOptions op;

  op.thermo() = kintera::ThermoOptions::from_yaml(filename);
  op.kinetics() = kintera::KineticsOptions::from_yaml(filename);

  auto config = YAML::LoadFile(filename);
  if (config["geometry"]) {
    op.coord() = CoordinateOptions::from_yaml(config["geometry"]);
  }

  // reconstruction
  if (config["reconstruct"]) {
    op.recon() = ReconstructOptions::from_yaml(config["reconstruct"], "scalar");
  }

  // riemann solver
  if (config["riemann"]) {
    op.riemann() = RiemannSolverOptions::from_yaml(config["riemann"]);
  }

  return op;
}

}  // namespace snap
