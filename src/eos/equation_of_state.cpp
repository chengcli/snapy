// yaml
#include <yaml-cpp/yaml.h>

// snap
#include <snap/snap.h>  // Index

#include <snap/utils/pull_neighbors.hpp>

#include "equation_of_state.hpp"

namespace snap {

EquationOfStateOptions EquationOfStateOptions::from_yaml(
    YAML::Node const& node) {
  EquationOfStateOptions op;

  op.type() = node["type"].as<std::string>("moist-mixture");
  op.density_floor() = node["density-floor"].as<double>(1.e-6);
  op.pressure_floor() = node["pressure-floor"].as<double>(1.e-3);
  op.temperature_floor() = node["temperature-floor"].as<double>(20.);
  op.limiter() = node["limiter"].as<bool>(false);
  op.eos_file() = node["eos-file"].as<std::string>("");

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

void EquationOfStateImpl::apply_conserved_limiter_(torch::Tensor const& cons) {
  if (!options.limiter()) return;  // no limiter
  cons.masked_fill_(torch::isnan(cons), 0.);
  cons[IDN].clamp_min_(options.density_floor());

  auto nghost = pcoord->options.nghost();
  auto interior = get_interior(cons.sizes(), nghost);
  int nvapor = options.thermo().vapor_ids().size() - 1;
  int ncloud = options.thermo().cloud_ids().size();
  // for (int i = ICY; i < ICY + nvapor; ++i)
  //   cons.index(interior)[i] = pull_neighbors3(cons.index(interior)[i]);
  //  batched
  cons.index(interior).narrow(0, ICY, nvapor) =
      pull_neighbors4(cons.index(interior).narrow(0, ICY, nvapor));
  cons.narrow(0, ICY + nvapor, ncloud).clamp_min_(0.);

  auto mom = cons.narrow(0, IVX, 3).clone();
  pcoord->vec_raise_(mom);

  auto ke = 0.5 * (mom * cons.narrow(0, IVX, 3)).sum(0) / cons[IDN];
  auto min_temp = options.temperature_floor() * torch::ones_like(ke);
  auto min_ie = compute("UT->I", {cons, min_temp});
  cons[IPR].clamp_min_(ke + min_ie);
}

void EquationOfStateImpl::apply_primitive_limiter_(torch::Tensor const& prim) {
  if (!options.limiter()) return;  // no limiter
  prim.masked_fill_(torch::isnan(prim), 0.);
  prim[IDN].clamp_min_(options.density_floor());

  int ny = options.thermo().vapor_ids().size() +
           options.thermo().cloud_ids().size() - 1;
  prim.narrow(0, ICY, ny).clamp_min_(0.);

  prim[IPR].clamp_min_(options.pressure_floor());
}

}  // namespace snap
