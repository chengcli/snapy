// kintere
#include <kintera/constants.h>

#include <kintera/thermo/eval_uhs.hpp>
#include <kintera/thermo/thermo.hpp>

// snap
#include <snap/snap.h>

#include <snap/registry.hpp>

#include "equation_of_state.hpp"

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

  // populate buffers
  int nc1 = options.coord().nc1();
  int nc2 = options.coord().nc2();
  int nc3 = options.coord().nc3();

  _prim = register_buffer(
      "W", torch::empty({nvar(), nc3, nc2, nc1}, torch::kFloat64));

  _cons = register_buffer(
      "U", torch::empty({nvar(), nc3, nc2, nc1}, torch::kFloat64));

  _gamma = register_buffer("A", torch::empty({nc3, nc2, nc1}, torch::kFloat64));

  _cs = register_buffer("L", torch::empty({nc3, nc2, nc1}, torch::kFloat64));

  _ke = register_buffer("K", torch::empty({nc3, nc2, nc1}, torch::kFloat64));

  _ie = register_buffer("I", torch::empty({nc3, nc2, nc1}, torch::kFloat64));
}

torch::Tensor IdealGasImpl::compute(std::string ab,
                                    std::vector<torch::Tensor> const &args) {
  if (ab == "W->U") {
    _prim.set_(args[0]);
    _prim2cons(_prim, _cons);
    return _cons;
  } else if (ab == "W->I") {
    _prim.set_(args[0]);
    _prim2intEng(_prim, _ie);
    return _ie;
  } else if (ab == "U->W") {
    _cons.set_(args[0]);
    _cons2prim(_cons, _prim);
    return _prim;
  } else if (ab == "W->A") {
    auto gammad =
        (pthermo->options.cref_R()[0] + 1) / pthermo->options.cref_R()[0];
    _gamma.set_(gammad * torch::ones_like(args[0][Index::IDN]));
    return _gamma;
  } else if (ab == "WA->L") {
    auto dens = args[0][Index::IDN];
    auto pres = args[0][Index::IPR];
    _cs.set_(torch::sqrt(args[1] * pres / dens));
    return _cs;
  } else {
    TORCH_CHECK(false, "Unknown abbreviation: ", ab);
  }
}

void IdealGasImpl::_prim2intEng(torch::Tensor prim, torch::Tensor &ie) {
  auto gammad =
      (pthermo->options.cref_R()[0] + 1) / pthermo->options.cref_R()[0];
  ie = prim[Index::IPR] / (gammad - 1);
}

void IdealGasImpl::_prim2cons(torch::Tensor prim, torch::Tensor &cons) {
  _apply_primitive_limiter_(prim);

  // den -> den
  cons[Index::IDN] = prim[Index::IDN];

  // vel -> mom
  auto out = cons.narrow(0, Index::IVX, 3);
  torch::mul_out(out, prim.narrow(0, Index::IVX, 3), prim[Index::IDN]);

  pcoord->vec_lower_(cons);

  // KE
  _ke.set_(
      (prim.narrow(0, Index::IVX, 3) * cons.narrow(0, Index::IVX, 3)).sum(0));
  _ke *= 0.5;

  // IE
  _prim2intEng(prim, _ie);

  out = cons[Index::IPR];
  torch::add_out(out, _ke, _ie);

  _apply_conserved_limiter_(cons);
}

void IdealGasImpl::_cons2prim(torch::Tensor cons, torch::Tensor &prim) {
  _apply_conserved_limiter_(cons);

  // den -> den
  prim[Index::IDN] = cons[Index::IDN];

  // mom -> vel
  auto out = prim.narrow(0, Index::IVX, 3);
  torch::div_out(out, cons.narrow(0, Index::IVX, 3), prim[Index::IDN]);

  pcoord->vec_raise_(prim);

  // KE (TODO: cli, new kernel for this operation)
  _ke.set_(
      (prim.narrow(0, Index::IVX, 3) * cons.narrow(0, Index::IVX, 3)).sum(0));
  _ke *= 0.5;

  torch::sub_out(_ie, cons[Index::IPR], _ke);

  // eng -> pr
  auto gammad =
      (pthermo->options.cref_R()[0] + 1) / pthermo->options.cref_R()[0];

  prim[Index::IPR] = (gammad - 1) * _ie;

  _apply_primitive_limiter_(prim);
}

}  // namespace snap
