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
  } else {
    TORCH_WARN("no geometry specified, using default coordinate model");
  }

  // project primitive variables
  op.proj() = PrimitiveProjectorOptions::from_yaml(config);

  if (!config["dynamics"]) {
    TORCH_WARN("no dynamics specified, using default hydro model");
    return op;
  }

  auto dyn = config["dynamics"];

  // equation of state
  if (dyn["equation-of-state"]) {
    op.eos() = EquationOfStateOptions::from_yaml(dyn["equation-of-state"]);
  } else {
    TORCH_WARN("no equation of state specified, using default EOS model");
  }

  op.eos().coord() = op.coord();
  op.eos().thermo() = op.thermo();

  // reconstruction
  if (dyn["reconstruct"]) {
    op.recon1() = ReconstructOptions::from_yaml(dyn["reconstruct"], "vertical");
    op.recon23() =
        ReconstructOptions::from_yaml(dyn["reconstruct"], "horizontal");
  } else {
    TORCH_WARN("no reconstruction specified, using default recon model");
  }

  // riemann solver
  if (dyn["riemann-solver"]) {
    op.riemann() = RiemannSolverOptions::from_yaml(dyn["riemann-solver"]);
  } else {
    TORCH_WARN("no riemann solver specified, using default riemann model");
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
