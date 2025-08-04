// kintere
#include <kintera/constants.h>

#include <kintera/thermo/thermo.hpp>

// snap
#include <snap/snap.h>

#include <snap/registry.hpp>

#include "eos_dispatch.hpp"
#include "ideal_gas.hpp"

namespace snap {

IdealGasImpl::IdealGasImpl(EquationOfStateOptions const &options_)
    : EquationOfStateImpl(options_) {
  reset();
}

void IdealGasImpl::reset() {
  // set up coordinate model
  pcoord = register_module_op(this, "coord", options.coord());

  // set up thermodynamics model
  pthermo = register_module("thermo", kintera::ThermoY(options.thermo()));
}

torch::Tensor IdealGasImpl::compute(std::string ab,
                                    std::vector<torch::Tensor> const &args) {
  if (ab == "W->U") {
    auto w = args[0];
    auto u = args.size() > 1 ? args[1] : torch::empty_like(w);
    _prim2cons(w, u);
    return u;
  } else if (ab == "U->W") {
    auto u = args[0];
    auto w = args.size() > 1 ? args[1] : torch::empty_like(u);
    _cons2prim(u, w);
    return w;
  } else if (ab == "W->I") {
    auto w = args[0];
    return _prim2intEng(w);
  } else if (ab == "W->T") {
    auto w = args[0];
    auto Rd = kintera::constants::Rgas / kintera::species_weights[0];
    return w[IPR] / (w[IDN] * Rd);
  } else if (ab == "UT->I") {
    auto w = args[0];
    auto temp = args[1];
    return _temp2intEng(w, temp);
  } else if (ab == "W->A") {
    auto w = args[0];
    auto gammad =
        (pthermo->options.cref_R()[0] + 1) / pthermo->options.cref_R()[0];
    return gammad * torch::ones_like(w[IDN]);
  } else if (ab == "WA->L") {
    auto w = args[0];
    auto gamma = args[1];
    auto dens = w[IDN];
    auto pres = w[IPR];
    return torch::sqrt(gamma * pres / dens);
  } else {
    TORCH_CHECK(false, "Unknown abbreviation: ", ab);
  }
}

void IdealGasImpl::_prim2cons(torch::Tensor prim, torch::Tensor &cons) {
  apply_primitive_limiter_(prim);

  // den -> den
  cons[IDN] = prim[IDN];

  // vel -> mom
  auto out = cons.narrow(0, IVX, 3);
  torch::mul_out(out, prim.narrow(0, IVX, 3), prim[IDN]);

  pcoord->vec_lower_(out);

  // KE
  auto ke = 0.5 * (prim.narrow(0, IVX, 3) * cons.narrow(0, IVX, 3)).sum(0);

  // IE
  cons[IPR] = _prim2intEng(prim);
  cons[IPR] += ke;

  apply_conserved_limiter_(cons);
}

void IdealGasImpl::_cons2prim(torch::Tensor cons, torch::Tensor &prim) {
  apply_conserved_limiter_(cons);

  auto gammad =
      (pthermo->options.cref_R()[0] + 1) / pthermo->options.cref_R()[0];

  auto iter = at::TensorIteratorConfig()
                  .resize_outputs(false)
                  .check_all_same_dtype(false)
                  .declare_static_shape(prim.sizes(), /*squash_dims=*/0)
                  .add_output(prim)
                  .add_input(cons)
                  .build();

  at::native::ideal_gas_cons2prim(cons.device().type(), iter, gammad);

  apply_primitive_limiter_(prim);
}

torch::Tensor IdealGasImpl::_prim2intEng(torch::Tensor prim) {
  auto gammad =
      (pthermo->options.cref_R()[0] + 1) / pthermo->options.cref_R()[0];
  return prim[IPR] / (gammad - 1);
}

torch::Tensor IdealGasImpl::_temp2intEng(torch::Tensor cons,
                                         torch::Tensor temp) {
  auto mud = kintera::species_weights[0];
  auto Rd = kintera::constants::Rgas / mud;
  auto cvd = kintera::species_cref_R[0] * Rd;
  return cons[IDN] * cvd * temp;
}

}  // namespace snap
