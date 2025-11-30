// yaml
#include <yaml-cpp/yaml.h>

// snap
#include "scalar.hpp"

namespace snap {

ScalarOptions ScalarOptionsImpl::from_yaml(std::string const& filename) {
  auto op = ScalarOptionsImpl::create();

  op->thermo() = kintera::ThermoOptionsImpl::from_yaml(filename);
  op->kinetics() = kintera::KineticsOptionsImpl::from_yaml(filename);

  return op;
}

}  // namespace snap
