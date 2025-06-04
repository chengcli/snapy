// base
#include <configure.h>

// snap
#include <snap/snap.h>

#include <snap/registry.hpp>

#include "eos_formatter.hpp"
#include "equation_of_state.hpp"

namespace snap {
void call_ideal_gas_cpu(at::TensorIterator& iter);
__attribute__((weak)) void call_ideal_gas_cuda(at::TensorIterator& iter) {}

void IdealGasImpl::reset() {
  if (options.thermo().vapor_ids().size() != 0 ||
      options.thermo().cloud_ids().size() != 0) {
    std::stringstream msg;
    msg << "IdealGasEOS should not have vapor or cloud";
    throw std::runtime_error(msg.str());
  }

  // set up thermodynamics model
  pthermo = register_module("thermo", kintera::ThermoY(options.thermo()));

  // set up coordinate model
  pcoord = register_module_op(this, "coord", options.coord());
}

void IdealGasImpl::cons2prim(torch::Tensor prim, torch::Tensor cons) const {
  if (!HAS_SHARED("hydro/gammad")) {
    // SET_SHARED("hydro/gammad") = pthermo->get_gammad(cons, kConserved);
    SET_SHARED("hydro/gammad") =
        pthermo->options.gammad() * torch::ones_like(cons[Index::IDN]);
  }

  auto iter = at::TensorIteratorConfig()
                  .resize_outputs(false)
                  .check_all_same_dtype(true)
                  .declare_static_shape(cons.sizes(), /*squash_dims=*/0)
                  .add_output(prim)
                  .add_input(cons)
                  .add_input(GET_SHARED("hydro/gammad"))
                  .build();

  if (cons.is_cpu()) {
    call_ideal_gas_cpu(iter);
  } else if (cons.is_cuda()) {
    call_ideal_gas_cuda(iter);
  } else {
    cons2prim_fallback(prim, cons, GET_SHARED("hydro/gammad"));
  }

  if (options.limiter()) {
    _apply_primitive_limiter_inplace(prim);
    prim2cons(cons, prim);
  }
}

void IdealGasImpl::cons2prim_fallback(torch::Tensor prim, torch::Tensor cons,
                                      torch::Tensor gammad) const {
  //_apply_conserved_limiter_inplace(cons);

  // den -> den
  prim[Index::IDN] = cons[Index::IDN];

  // mom -> vel
  prim.slice(0, 1, Index::IPR) =
      cons.slice(0, 1, Index::IPR) / prim[Index::IDN];

  pcoord->vec_raise_inplace(prim);

  auto ke =
      0.5 *
      (prim.narrow(0, Index::IVX, 3) * cons.narrow(0, Index::IVX, 3)).sum(0);

  // eng -> pr
  prim[Index::IPR] = (gammad - 1.) * (cons[Index::IPR] - ke);
}

void IdealGasImpl::prim2cons(torch::Tensor cons, torch::Tensor prim) const {
  //_apply_primitive_limiter_inplace(prim);

  // den -> den
  cons[Index::IDN] = prim[Index::IDN];

  // vel -> mom
  cons.slice(0, 1, Index::IPR) =
      prim.slice(0, 1, Index::IPR) * prim[Index::IDN];

  pcoord->vec_lower_inplace(cons);

  auto ke =
      0.5 *
      (prim.narrow(0, Index::IVX, 3) * cons.narrow(0, Index::IVX, 3)).sum(0);

  // pr -> eng
  // auto gammad = pthermo->get_gammad(cons, kConserved);
  auto gammad = pthermo->options.gammad();
  cons[Index::IPR] = prim[Index::IPR] / (gammad - 1.) + ke;

  _apply_conserved_limiter_inplace(cons);
}

torch::Tensor IdealGasImpl::sound_speed(torch::Tensor prim) const {
  // auto gammad = pthermo->get_gammad(prim);
  auto gammad = pthermo->options.gammad();
  return torch::sqrt(gammad * prim[Index::IPR] / prim[Index::IDN]);
}

}  // namespace snap
