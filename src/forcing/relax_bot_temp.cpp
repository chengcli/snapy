// yaml
#include <yaml-cpp/yaml.h>

// snap
#include <snap/snap.h>

#include <snap/hydro/hydro.hpp>
#include <snap/mesh/meshblock.hpp>

#include "forcing.hpp"

namespace snap {

RelaxBotTempOptions RelaxBotTempOptionsImpl::from_yaml(
    YAML::Node const& forcing) {
  if (!forcing["relax-bot-temp"]) return nullptr;

  auto node = forcing["relax-bot-temp"];
  auto op = RelaxBotTempOptionsImpl::create();

  op->tau() = node["tau"].as<double>(0.0);
  op->btemp() = node["btemp"].as<double>(300.0);

  TORCH_CHECK(op->tau() > 0.,
              "RelaxBotTempOptions: tau must be greater than zero.");

  return op;
}

RelaxBotTempImpl::RelaxBotTempImpl(RelaxBotTempOptions const& options_,
                                   torch::nn::Module* p)
    : options(options_) {
  phydro = dynamic_cast<HydroImpl const*>(p);
  reset();
}

void RelaxBotTempImpl::reset() {
  TORCH_CHECK(phydro, "[RelaxBotTemp] Parent Hydro is null");
}

torch::Tensor RelaxBotTempImpl::forward(torch::Tensor du, torch::Tensor w,
                                        torch::Tensor temp, double dt) {
  // Applies at the physical lower x1 boundary only: under an x1-decomposed
  // layout (nb1 > 1) a rank whose lower face is an internal block interface
  // must not force there.
  if (!phydro->pmb->options->is_physical_boundary(0, 0, -1)) return du;

  auto bottom = phydro->pmb->part(
      {0, 0, -1}, PartOptions().exterior(false).depth(1).ndim(3));
  auto rho = w[IDN].index(bottom);
  auto temp_bot = temp.index(bottom);
  auto cv = phydro->peos->specific_heat_cv(w, temp).index(bottom);
  du[IPR].index(bottom) +=
      dt / options->tau() * rho * cv * (options->btemp() - temp_bot);
  return du;
}

}  // namespace snap
