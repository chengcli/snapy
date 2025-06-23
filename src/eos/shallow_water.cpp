// snap
#include <snap/snap.h>

#include <snap/registry.hpp>

#include "equation_of_state.hpp"

namespace snap {

ShallowWaterImpl::ShallowWaterImpl(EquationOfStateOptions const &options_)
    : EquationOfStateImpl(options_) {
  reset();
}

void ShallowWaterImpl::reset() {
  // set up coordinate model
  pcoord = register_module_op(this, "coord", options.coord());

  // populate buffers
  int nx1 = options.coord().nx1();
  int nx2 = options.coord().nx2();
  int nx3 = options.coord().nx3();

  _prim = register_buffer(
      "prim", torch::empty({nhydro(), nx3, nx2, nx1}, torch::kFloat64));

  _cons = register_buffer(
      "cons", torch::empty({nhydro(), nx3, nx2, nx1}, torch::kFloat64));

  _cs = register_buffer("cs", torch::empty({nx3, nx2, nx1}, torch::kFloat64));
}

torch::Tensor const &ShallowWaterImpl::compute(
    std::string ab, std::vector<torch::Tensor> const &args) {
  if (ab == "prim->cons") {
    _prim.set_(args[0]);
    _prim2cons(_prim, _cons);
    return _cons;
  } else if (ab == "cons->prim") {
    _cons.set_(args[0]);
    _cons2prim(_cons, _prim);
    return _prim;
  } else if (ab == "prim->cs") {
    _prim.set_(args[0]);
    _sound_speed(args[0], _cs);
    return _cs;
  } else {
    TORCH_CHECK(false, "Unknown abbreviation: ", ab);
  }
}

void ShallowWaterImpl::_cons2prim(torch::Tensor cons,
                                  torch::Tensor &prim) const {
  _apply_conserved_limiter_(cons);

  prim[Index::IDN] = cons[Index::IDN];
  prim.narrow(0, Index::IVX, 3) =
      cons.narrow(0, Index::IVX, 3) / cons[Index::IDN];

  pcoord->vec_raise_(prim);

  _apply_primitive_limiter_(prim);
}

void ShallowWaterImpl::_prim2cons(torch::Tensor prim,
                                  torch::Tensor &cons) const {
  _apply_primitive_limiter_(prim);

  cons[Index::IDN] = prim[Index::IDN];
  cons.narrow(0, Index::IVX, 3) =
      prim.narrow(0, Index::IVX, 3) * prim[Index::IDN];

  pcoord->vec_lower_(cons);

  _apply_conserved_limiter_(cons);
}

torch::Tensor ShallowWaterImpl::_sound_speed(torch::Tensor prim,
                                             torch::Tensor &out) const {
  return torch::sqrt(prim[Index::IDN]);
}

}  // namespace snap
