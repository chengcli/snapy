// torch
#include <torch/torch.h>

// base
#include <configure.h>

// snap
#include <snap/snap.h>

#include "eos_formatter.hpp"
#include "equation_of_state.hpp"

namespace snap {
void ShallowWaterImpl::reset() {
  TORCH_CHECK(options.thermo().nvapor() == 0,
              "ShallowWaterEOS should not have vapor");

  TORCH_CHECK(options.thermo().ncloud() == 0,
              "ShallowWaterEOS should not have cloud");

  // set up thermodynamics model
  pthermo = register_module("thermo", Thermodynamics(options.thermo()));

  // set up coordinate model
  pcoord = register_module_op(this, "coord", options.coord());
}

void ShallowWaterImpl::cons2prim(torch::Tensor prim, torch::Tensor cons) const {
  _apply_conserved_limiter_inplace(cons);

  prim[Index::IDN] = cons[Index::IDN];
  prim.narrow(0, Index::IVX, 3) =
      cons.narrow(0, Index::IVX, 3) / cons[Index::IDN];

  pcoord->vec_raise_inplace(prim);

  _apply_primitive_limiter_inplace(prim);
}

void ShallowWaterImpl::prim2cons(torch::Tensor cons, torch::Tensor prim) const {
  _apply_primitive_limiter_inplace(prim);

  cons[Index::IDN] = prim[Index::IDN];
  cons.narrow(0, Index::IVX, 3) =
      prim.narrow(0, Index::IVX, 3) * prim[Index::IDN];

  pcoord->vec_lower_inplace(cons);

  _apply_conserved_limiter_inplace(cons);
}

torch::Tensor ShallowWaterImpl::sound_speed(torch::Tensor prim) const {
  return torch::sqrt(prim[Index::IDN]);
}
}  // namespace snap
