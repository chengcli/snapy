// torch
#include <torch/torch.h>

// snap
#include <snap/snap.h>

#include <snap/registry.hpp>

#include "eos_formatter.hpp"
#include "equation_of_state.hpp"

namespace snap {

EquationOfStateOptions::EquationOfStateOptions(ParameterInput pin) {
  type(pin->GetOrAddString("hydro", "eos", "ideal_gas"));
  // thermo(ThermodynamicsOptions(pin));
  coord(CoordinateOptions(pin));
}

void EquationOfStateImpl::_apply_conserved_limiter_inplace(
    torch::Tensor& cons) const {
  cons.narrow(0, Index::IVX, 3)
      .masked_fill_(torch::isnan(cons.narrow(0, Index::IVX, 3)), 0.);
}

void EquationOfStateImpl::_apply_primitive_limiter_inplace(
    torch::Tensor& prim) const {
  prim[Index::IDN].clamp_min_(options.density_floor());
  prim[Index::IPR].clamp_min_(options.pressure_floor());
}

}  // namespace snap
