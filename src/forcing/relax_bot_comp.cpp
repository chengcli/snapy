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
  auto wbot = w.index(bottom);
  int ny = pthermo_y->options->species().size() - 1;

  auto target_w = wbot.clone();
  auto target_x = pthermo_y->compute("Y->X", {target_w.narrow(0, ICY, ny)});
  for (int n = 0; n < species_ids.size(); ++n) {
    target_x.select(-1, species_ids[n]).fill_(options->xfrac()[n]);
  }

  auto dry = 1. - target_x.narrow(-1, 1, ny).sum(-1);
  TORCH_CHECK(dry.min().item<double>() >= -1.e-12,
              "[RelaxBotComp] Configured and retained species leave a "
              "negative dry mole fraction.");
  target_x.select(-1, 0) = dry.clamp_min(0.);

  auto target_y = pthermo_x->compute("X->Y", {target_x});
  target_w.narrow(0, ICY, ny) = target_y;
  auto ivol = pthermo_y->compute("DY->V", {target_w[IDN], target_y});
  target_w[IPR] = pthermo_y->compute("VT->P", {ivol, temp.index(bottom3)});

  auto target_u = phydro->peos->compute("W->U", {target_w});
  auto current_u = phydro->peos->compute("W->U", {wbot});
  du.index(bottom) += dt / options->tau() * (target_u - current_u);
  return du;
}

}  // namespace snap
