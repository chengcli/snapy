// yaml
#include <yaml-cpp/yaml.h>

// snap
#include <snap/snap.h>  // Index

#include <snap/eos/aneos.hpp>
#include <snap/eos/ideal_gas.hpp>
#include <snap/eos/ideal_moist.hpp>
#include <snap/eos/moist_mixture.hpp>
#include <snap/eos/plume_eos.hpp>
#include <snap/eos/shallow_water.hpp>
#include <snap/utils/pull_neighbors.hpp>

#include "equation_of_state.hpp"

namespace snap {

EquationOfStateOptions EquationOfStateOptionsImpl::from_yaml(
    std::string const& filename, bool verbose) {
  auto config = YAML::LoadFile(filename);
  auto op = EquationOfStateOptionsImpl::create();

  if (!config["dynamics"]) return op;
  if (!config["dynamics"]["equation-of-state"]) return op;

  auto node = config["dynamics"]["equation-of-state"];

  op->type() = node["type"].as<std::string>("moist-mixture");
  op->density_floor() = node["density-floor"].as<double>(1.e-6);
  op->pressure_floor() = node["pressure-floor"].as<double>(1.e-3);
  op->temperature_floor() = node["temperature-floor"].as<double>(20.);
  op->limiter() = node["limiter"].as<bool>(false);
  op->eos_file() = node["eos-file"].as<std::string>("");

  op->thermo() = kintera::ThermoOptionsImpl::from_yaml(filename, verbose);

  if (op->thermo()) {
    TORCH_CHECK(
        NMASS == 0 || (op->thermo()->vapor_ids().size() +
                           op->thermo()->cloud_ids().size() ==
                       1 + NMASS),
        "Athena++ style indexing is enabled (NMASS > 0), but the number of "
        "vapor and cloud species in the thermodynamics options does not match "
        "the expected number of vapor + cloud species = ",
        1 + NMASS);
  }

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
  if (!options->limiter()) return;  // no limiter
  cons.masked_fill_(torch::isnan(cons), 0.);
  cons[IDN].clamp_min_(options->density_floor());

  auto nghost = pcoord->options->nghost();
  auto interior = get_interior(cons.sizes(), nghost);
  int nvapor = options->thermo()->vapor_ids().size() - 1;
  int ncloud = options->thermo()->cloud_ids().size();
  // for (int i = ICY; i < ICY + nvapor; ++i)
  //   cons.index(interior)[i] = pull_neighbors3(cons.index(interior)[i]);
  //  batched
  cons.index(interior).narrow(0, ICY, nvapor) =
      pull_neighbors4(cons.index(interior).narrow(0, ICY, nvapor));
  cons.narrow(0, ICY + nvapor, ncloud).clamp_min_(0.);

  auto mom = cons.narrow(0, IVX, 3).clone();
  pcoord->vec_raise_(mom);

  auto ke = 0.5 * (mom * cons.narrow(0, IVX, 3)).sum(0) / cons[IDN];
  auto min_temp = options->temperature_floor() * torch::ones_like(ke);
  auto min_ie = compute("UT->I", {cons, min_temp});
  cons[IPR].clamp_min_(ke + min_ie);
}

void EquationOfStateImpl::apply_primitive_limiter_(torch::Tensor const& prim) {
  if (!options->limiter()) return;  // no limiter
  prim.masked_fill_(torch::isnan(prim), 0.);
  prim[IDN].clamp_min_(options->density_floor());

  int ny = options->thermo()->vapor_ids().size() +
           options->thermo()->cloud_ids().size() - 1;
  prim.narrow(0, ICY, ny).clamp_min_(0.);

  prim[IPR].clamp_min_(options->pressure_floor());
}

EquationOfState EquationOfStateImpl::create(EquationOfStateOptions const& opts,
                                            torch::nn::Module* p,
                                            std::string const& name) {
  TORCH_CHECK(p, "[EquationOfState] Parent module pointer is null.");
  TORCH_CHECK(opts, "[EquationOfState] Options pointer is null.");

  if (opts->type() == "ideal-gas") {
    return p->register_module(name, IdealGas(opts));
  } else if (opts->type() == "ideal-moist") {
    return p->register_module(name, IdealMoist(opts));
  } else if (opts->type() == "moist-mixture") {
    return p->register_module(name, MoistMixture(opts));
  } else if (opts->type() == "aneos") {
    return p->register_module(name, ANEOS(opts));
  } else if (opts->type() == "shallow-water") {
    return p->register_module(name, ShallowWater(opts));
  } else if (opts->type() == "plume-eos") {
    return p->register_module(name, PlumeEOS(opts));
  } else {
    TORCH_CHECK(false, "EquationOfState: Unknown type: ", opts->type());
  }
}

}  // namespace snap
