// snap
#include "shallow_water.hpp"

#include <snap/snap.h>

#include <snap/registry.hpp>

namespace snap {

ShallowWaterImpl::ShallowWaterImpl(EquationOfStateOptions const &options_)
    : EquationOfStateImpl(options_) {
  reset();
}

void ShallowWaterImpl::reset() {
  // set up coordinate model
  pcoord = register_module_op(this, "coord", options.coord());

  // populate buffers
  int nc1 = options.coord().nc1();
  int nc2 = options.coord().nc2();
  int nc3 = options.coord().nc3();

  _prim = register_buffer(
      "prim", torch::empty({nvar(), nc3, nc2, nc1}, torch::kFloat64));

  _cons = register_buffer(
      "cons", torch::empty({nvar(), nc3, nc2, nc1}, torch::kFloat64));

  _cs = register_buffer("cs", torch::empty({nc3, nc2, nc1}, torch::kFloat64));
}

torch::Tensor ShallowWaterImpl::compute(
    std::string ab, std::vector<torch::Tensor> const &args) {
  if (ab == "W->U") {
    _prim.set_(args[0]);
    _prim2cons(_prim, _cons);
    return _cons;
  } else if (ab == "U->W") {
    _cons.set_(args[0]);
    _cons2prim(_cons, _prim);
    return _prim;
  } else if (ab == "W->A") {
    return torch::Tensor();
  } else if (ab == "WA->L") {
    _prim.set_(args[0]);
    _gravity_wave_speed(args[0], _cs);
    return _cs;
  } else {
    TORCH_CHECK(false, "Unknown abbreviation: ", ab);
  }
}

void ShallowWaterImpl::_cons2prim(torch::Tensor cons, torch::Tensor &prim) {
  apply_conserved_limiter_(cons);

  prim[Index::IDN] = cons[Index::IDN];

  // lvalue view
  auto out = prim.narrow(0, Index::IVX, 3);
  torch::div_out(out, cons.narrow(0, Index::IVX, 3), cons[Index::IDN]);

  pcoord->vec_raise_(prim);

  apply_primitive_limiter_(prim);
}

void ShallowWaterImpl::_prim2cons(torch::Tensor prim, torch::Tensor &cons) {
  apply_primitive_limiter_(prim);

  cons[Index::IDN] = prim[Index::IDN];

  // lvalue view
  auto out = cons.narrow(0, Index::IVX, 3);
  torch::mul_out(out, prim.narrow(0, Index::IVX, 3), prim[Index::IDN]);

  pcoord->vec_lower_(cons);

  apply_conserved_limiter_(cons);
}

void ShallowWaterImpl::_gravity_wave_speed(torch::Tensor prim,
                                           torch::Tensor &out) const {
  torch::sqrt_out(out, prim[Index::IDN]);
}

}  // namespace snap
