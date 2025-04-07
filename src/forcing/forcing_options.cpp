// base
#include <input/parameter_input.hpp>

// snap
#include "forcing.hpp"

namespace snap {
ConstGravityOptions::ConstGravityOptions(ParameterInput pin) {
  grav1(pin->GetOrAddReal("forcing", "grav1", 0.));
  grav2(pin->GetOrAddReal("forcing", "grav2", 0.));
  grav3(pin->GetOrAddReal("forcing", "grav3", 0.));
}

CoriolisOptions::CoriolisOptions(ParameterInput pin) {
  omega1(pin->GetOrAddReal("forcing", "omega1", 0.));
  omega2(pin->GetOrAddReal("forcing", "omega2", 0.));
  omega3(pin->GetOrAddReal("forcing", "omega3", 0.));

  omegax(pin->GetOrAddReal("forcing", "omegax", 0.));
  omegay(pin->GetOrAddReal("forcing", "omegay", 0.));
  omegaz(pin->GetOrAddReal("forcing", "omegaz", 0.));
}
}  // namespace snap
