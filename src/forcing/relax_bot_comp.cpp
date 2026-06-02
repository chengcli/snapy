// C/C++
#include <algorithm>
#include <numeric>
#include <unordered_set>

// yaml
#include <yaml-cpp/yaml.h>

// snap
#include <snap/snap.h>

#include <snap/hydro/hydro.hpp>
#include <snap/mesh/meshblock.hpp>

#include "forcing.hpp"

namespace snap {

RelaxBotCompOptions RelaxBotCompOptionsImpl::from_yaml(
    YAML::Node const& forcing) {
  if (!forcing["relax-bot-comp"]) return nullptr;

  auto node = forcing["relax-bot-comp"];
  auto op = RelaxBotCompOptionsImpl::create();

  op->tau() = node["tau"].as<double>(0.0);
  op->species() =
      node["species"].as<std::vector<std::string>>(std::vector<std::string>{});
  op->xfrac() = node["xfrac"].as<std::vector<double>>(std::vector<double>{});

  TORCH_CHECK(
      op->species().size() == op->xfrac().size(),
      "RelaxBotCompOptions: 'species' and 'xfrac' must have the same length.");
  TORCH_CHECK(op->tau() > 0.,
              "RelaxBotCompOptions: tau must be greater than zero.");

  std::unordered_set<std::string> names;
  for (int n = 0; n < op->species().size(); ++n) {
    TORCH_CHECK(names.insert(op->species()[n]).second,
                "RelaxBotCompOptions: duplicate species '", op->species()[n],
                "'.");
    TORCH_CHECK(op->xfrac()[n] >= 0. && op->xfrac()[n] <= 1.,
                "RelaxBotCompOptions: xfrac values must lie in [0, 1].");
  }
  TORCH_CHECK(std::accumulate(op->xfrac().begin(), op->xfrac().end(), 0.) <= 1.,
              "RelaxBotCompOptions: configured xfrac values must sum to no "
              "more than one.");

  return op;
}

RelaxBotCompImpl::RelaxBotCompImpl(RelaxBotCompOptions const& options_,
                                   torch::nn::Module* p)
    : options(options_) {
  phydro = dynamic_cast<HydroImpl const*>(p);
  reset();
}

void RelaxBotCompImpl::reset() {
  TORCH_CHECK(phydro, "[RelaxBotComp] Parent Hydro is null");
  auto thermo = phydro->options->eos()->thermo();
  TORCH_CHECK(thermo, "[RelaxBotComp] Thermodynamics options are required.");

  pthermo_y = register_module("thermo-y", kintera::ThermoY(thermo));
  pthermo_x = register_module("thermo-x", kintera::ThermoX(thermo));

  species_ids.clear();
  auto species = thermo->species();
  for (auto const& name : options->species()) {
    auto it = std::find(species.begin(), species.end(), name);
    TORCH_CHECK(it != species.end(), "[RelaxBotComp] Unknown species '", name,
                "'.");
    auto id = std::distance(species.begin(), it);
    TORCH_CHECK(id != 0, "[RelaxBotComp] Dry species '", name,
                "' cannot be configured directly.");
    species_ids.push_back(id);
  }
}

torch::Tensor RelaxBotCompImpl::forward(torch::Tensor du, torch::Tensor w,
                                        torch::Tensor temp, double dt) {
  auto bottom =
      phydro->pmb->part({0, 0, -1}, PartOptions().exterior(false).depth(1));
  auto bottom3 = phydro->pmb->part(
      {0, 0, -1}, PartOptions().exterior(false).depth(1).ndim(3));
  int ny = pthermo_y->options->species().size() - 1;

  auto wbot = w.index(bottom);
  auto rho = wbot[IDN];
  auto current_y = wbot.narrow(0, ICY, ny);
  auto target_x = pthermo_y->compute("Y->X", {current_y});
  for (int n = 0; n < species_ids.size(); ++n) {
    target_x.select(-1, species_ids[n]).fill_(options->xfrac()[n]);
  }

  auto dry = 1. - target_x.narrow(-1, 1, ny).sum(-1);
  target_x.select(-1, 0) = dry.clamp_min(0.);

  auto target_y = pthermo_x->compute("X->Y", {target_x});
  auto target_v = pthermo_y->compute("DY->V", {rho, target_y});
  auto current_v = pthermo_y->compute("DY->V", {rho, current_y});
  auto temp_bot = temp.index(bottom3);
  auto target_ie = pthermo_y->compute("VT->U", {target_v, temp_bot});
  auto current_ie = pthermo_y->compute("VT->U", {current_v, temp_bot});

  auto delta = torch::zeros_like(wbot);
  delta[IDN] = -rho * (target_y - current_y).sum(0);
  delta.narrow(0, ICY, ny) = rho * (target_y - current_y);
  delta[IPR] = target_ie - current_ie;
  du.index(bottom) += dt / options->tau() * delta;
  return du;
}

}  // namespace snap
