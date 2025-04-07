// torch
#include <torch/torch.h>

// fvm
#include <fvm/index.h>

#include <fvm/registry.hpp>

#include "eos_formatter.hpp"
#include "equation_of_state.hpp"

namespace snap {

EquationOfStateOptions::EquationOfStateOptions(ParameterInput pin) {
  type(pin->GetOrAddString("hydro", "eos", "ideal_gas"));
  thermo(ThermodynamicsOptions(pin));
  coord(CoordinateOptions(pin));
}

void EquationOfStateImpl::_apply_conserved_limiter_inplace(
    torch::Tensor& cons) const {
  cons.narrow(0, index::IVX, 3)
      .masked_fill_(torch::isnan(cons.narrow(0, index::IVX, 3)), 0.);
}

void EquationOfStateImpl::_apply_primitive_limiter_inplace(
    torch::Tensor& prim) const {
  prim[index::IDN].clamp_min_(options.density_floor());
  prim[index::IPR].clamp_min_(options.pressure_floor());
}

}  // namespace snap
