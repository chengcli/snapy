// snap
#include "equation_of_state.hpp"

#include <snap/snap.h>  // Index

namespace snap {

EquationOfStateOptions::EquationOfStateOptions(ParameterInput pin) {
  type(pin->GetOrAddString("hydro", "eos", "moist_mixture"));
  coord(CoordinateOptions(pin));
}

torch::Tensor EquationOfStateImpl::compute(
    std::string ab, std::vector<torch::Tensor> const& args) {
  TORCH_CHECK(false, "EquationOfStateImpl::compute() is not implemented.",
              "Please use this method in a derived class.");
}

torch::Tensor EquationOfStateImpl::get_buffer(std::string) const {
  TORCH_CHECK(false, "EquationOfStateImpl::get_buffer() is not implemented.",
              "Please use this method in a derived class.");
}

torch::Tensor EquationOfStateImpl::forward(torch::Tensor cons,
                                           torch::optional<torch::Tensor> out) {
  auto prim = out.value_or(torch::empty_like(cons));
  return compute("cons->prim", {cons, prim});
}

void EquationOfStateImpl::_apply_conserved_limiter_(torch::Tensor& cons) const {
  cons.narrow(0, Index::IVX, 3)
      .masked_fill_(torch::isnan(cons.narrow(0, Index::IVX, 3)), 0.);
  int ny = nhydro() - 5;
  cons.narrow(0, Index::ICY, ny).clamp_min_(0.);
}

void EquationOfStateImpl::_apply_primitive_limiter_(torch::Tensor& prim) const {
  prim[Index::IDN].clamp_min_(options.density_floor());
  prim[Index::IPR].clamp_min_(options.pressure_floor());
  int ny = nhydro() - 5;
  prim.narrow(0, Index::ICY, ny).clamp_min_(0.);
}

}  // namespace snap
