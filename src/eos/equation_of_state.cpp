// yaml
#include <yaml-cpp/yaml.h>

// torch
#include <ATen/TensorIterator.h>

// snap
#include <snap/snap.h>

#include <snap/coord/coord_utils.hpp>
#include <snap/hydro/hydro.hpp>
#include <snap/mesh/meshblock.hpp>
#include <snap/utils/log.hpp>

#include "aneos.hpp"
#include "eos_dispatch.hpp"
#include "equation_of_state.hpp"
#include "ideal_gas.hpp"
#include "ideal_moist.hpp"
#include "moist_mixture.hpp"
#include "plume_eos.hpp"
#include "shallow_water.hpp"

namespace snap {

EquationOfStateOptions EquationOfStateOptionsImpl::from_yaml(
    std::string const& filename, bool verbose) {
  auto config = YAML::LoadFile(filename);
  auto op = EquationOfStateOptionsImpl::create();

  if (!config["dynamics"]) return op;
  if (!config["dynamics"]["equation-of-state"]) return op;

  auto node = config["dynamics"]["equation-of-state"];
  op->verbose() = node["verbose"].as<bool>(verbose);

  op->type() = node["type"].as<std::string>("moist-mixture");
  if (op->verbose()) {
    SINFO(EquationOfStateOptions) << "EOS type = " << op->type() << std::endl;
  }

  op->gammad() = node["gammad"].as<double>(1.4);
  op->weight() = node["weight"].as<double>(29.e-3);

  op->density_floor() = node["density-floor"].as<double>(1.e-6);
  if (op->verbose()) {
    SINFO(EquationOfStateOptions)
        << "density floor = " << op->density_floor() << std::endl;
  }

  op->pressure_floor() = node["pressure-floor"].as<double>(1.e-3);
  op->temperature_floor() = node["temperature-floor"].as<double>(20.);

  op->limiter() = node["limiter"].as<bool>(false);
  if (op->verbose()) {
    SINFO(EquationOfStateOptions)
        << "limiter = " << (op->limiter() ? "true" : "false") << std::endl;
  }

  op->eos_file() = node["eos-file"].as<std::string>("");
  if (op->verbose() && !op->eos_file().empty()) {
    SINFO(EquationOfStateOptions)
        << "eos file = " << op->eos_file() << std::endl;
  }

  op->thermo() = kintera::ThermoOptionsImpl::from_yaml(filename, op->verbose());

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

EquationOfStateImpl::EquationOfStateImpl(EquationOfStateOptions const& options_,
                                         torch::nn::Module* p)
    : options(options_) {
  phydro = dynamic_cast<HydroImpl const*>(p);
  TORCH_CHECK(phydro, "[EquationOfState] Parent module is null.");
}

torch::Tensor EquationOfStateImpl::compute(
    std::string ab, std::vector<torch::Tensor> const& args) {
  TORCH_CHECK(false, "[EquationOfState] compute() is not implemented.",
              "Please use this method in a derived class.");
}

/*torch::Tensor EquationOfStateImpl::get_buffer(std::string) const {
  TORCH_CHECK(false, "[EquationOfState] get_buffer() is not implemented.",
              "Please use this method in a derived class.");
}*/

torch::Tensor EquationOfStateImpl::forward(torch::Tensor cons,
                                           torch::optional<torch::Tensor> out) {
  auto prim = out.value_or(torch::empty_like(cons));
  return compute("U->W", {cons, prim});
}

void EquationOfStateImpl::apply_conserved_limiter_(torch::Tensor const& cons) {
  auto pmb = phydro->pmb;
  auto pcoord = pmb->pcoord;

  if (!options->limiter()) return;  // no limiter
  cons.masked_fill_(torch::isnan(cons), 0.);
  cons[IDN].clamp_min_(options->density_floor());

  // for (int i = ICY; i < ICY + nvapor; ++i)
  //   cons.index(interior)[i] = pull_neighbors3(cons.index(interior)[i]);
  //  batched
  // cons.index(interior).narrow(0, ICY, nvapor) =
  //    pull_neighbors4(cons.index(interior).narrow(0, ICY, nvapor));

  int ny = 0;
  if (options->thermo()) {
    auto nghost = pcoord->options->nghost();
    auto interior = pmb->part({0, 0, 0}, PartOptions().exterior(false));
    int nvapor = options->thermo()->vapor_ids().size() - 1;
    int ncloud = options->thermo()->cloud_ids().size();
    ny = nvapor + ncloud;

    auto vapor = cons.index(interior).narrow(0, ICY, nvapor);
    auto major = cons.index(interior)[IDN].unsqueeze(0);
    auto iter = at::TensorIteratorConfig()
                    .resize_outputs(false)
                    .declare_static_shape(vapor.sizes(),
                                          /*squash_dim=*/vapor.dim() - 1)
                    .add_output(vapor)
                    .add_owned_input(major.expand_as(vapor))
                    .build();

    int err = at::native::call_fix_vapor(cons.device().type(), iter);
    TORCH_CHECK(err == 0,
                "[EquationOfState] apply_conserved_limiter_: "
                "Failed to fix vapor mass fractions.");

    cons.narrow(0, ICY + nvapor, ncloud).clamp_min_(0.);
  }

  if (nvar() > IPR) {
    auto mom = cons.narrow(0, IVX, 3).clone();
    coord_vec_raise_(mom, pcoord->cosine_cell_kj);
    auto rho = cons[IDN] + cons.narrow(0, ICY, ny).sum(0);
    auto ke = 0.5 * (mom * cons.narrow(0, IVX, 3)).sum(0) / rho;
    auto min_temp = options->temperature_floor() * torch::ones_like(ke);
    auto min_ie = compute("UT->I", {cons, min_temp});
    cons[IPR].clamp_min_(ke + min_ie);
  }
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
    return p->register_module(name, IdealGas(opts, p));
  } else if (opts->type() == "ideal-moist") {
    return p->register_module(name, IdealMoist(opts, p));
  } else if (opts->type() == "moist-mixture") {
    return p->register_module(name, MoistMixture(opts, p));
  } else if (opts->type() == "aneos") {
    return p->register_module(name, ANEOS(opts, p));
  } else if (opts->type() == "shallow-water") {
    return p->register_module(name, ShallowWater(opts, p));
  } else if (opts->type() == "plume-eos") {
    return p->register_module(name, PlumeEOS(opts, p));
  } else {
    TORCH_CHECK(false, "EquationOfState: Unknown type: ", opts->type());
  }
}

}  // namespace snap
