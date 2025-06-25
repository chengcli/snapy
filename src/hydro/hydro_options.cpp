// yaml
#include <yaml-cpp/yaml.h>

// snap
#include "hydro.hpp"

namespace snap {

HydroOptions HydroOptions::from_yaml(std::string const& filename) {
  HydroOptions op;

  op.thermo() = kintera::ThermoOptions::from_yaml(filename);

  auto config = YAML::LoadFile(filename);
  if (config["geometry"]) {
    op.coord() = CoordinateOptions::from_yaml(config["geometry"]);
  }

  // project primitive variables
  op.proj() = PrimitiveProjectorOptions::from_yaml(config);

  // equation of state
  if (config["equation-of-state"]) {
    op.eos() = EquationOfStateOptions::from_yaml(config["equation-of-state"]);
  }
  op.eos().coord() = op.coord();
  op.eos().thermo() = op.thermo();

  // reconstruction
  if (config["reconstrct"]) {
    op.recon1() =
        ReconstructOptions::from_yaml(config["reconstruct"], "vertical");
    op.recon23() =
        ReconstructOptions::from_yaml(config["reconstruct"], "horizontal");
  }

  // riemann solver
  if (config["riemann"]) {
    op.riemann() = RiemannSolverOptions::from_yaml(config["riemann"]);
  }
  op.riemann().eos() = op.eos();

  // internal boundaries
  op.ib() = InternalBoundaryOptions::from_yaml(config);

  // implicit options
  op.vic() = ImplicitOptions::from_yaml(config);
  op.vic().recon() = op.recon1();
  op.vic().coord() = op.coord();

  return op;
}

}  // namespace snap
