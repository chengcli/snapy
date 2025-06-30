// yaml
#include <yaml-cpp/yaml.h>

// snap
#include <snap/snap.h>  // Index

#include "equation_of_state.hpp"

namespace snap {

EquationOfStateOptions EquationOfStateOptions::from_yaml(
    YAML::Node const& node) {
  EquationOfStateOptions op;

  op.type() = node["type"].as<std::string>("moist-mixture");
  op.density_floor() = node["density-floor"].as<double>(1.e-6);
  op.pressure_floor() = node["pressure-floor"].as<double>(1.e-3);
  op.limiter() = node["limiter"].as<bool>(false);

  return op;
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
  return compute("U->W", {cons, prim});
}

void EquationOfStateImpl::_apply_conserved_limiter_(torch::Tensor& cons) const {
  if (!options.limiter()) return;  // no limiter

  cons.narrow(0, Index::IVX, 3)
      .masked_fill_(torch::isnan(cons.narrow(0, Index::IVX, 3)), 0.);
  int ny = nvar() - 5;
  cons.narrow(0, Index::ICY, ny).clamp_min_(0.);
}

void EquationOfStateImpl::_apply_primitive_limiter_(torch::Tensor& prim) const {
  if (!options.limiter()) return;  // no limiter

  prim[Index::IDN].clamp_min_(options.density_floor());
  prim[Index::IPR].clamp_min_(options.pressure_floor());
  int ny = nvar() - 5;
  prim.narrow(0, Index::ICY, ny).clamp_min_(0.);
}

}  // namespace snap
