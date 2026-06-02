// yaml
#include <yaml-cpp/yaml.h>

// snap
#include <snap/snap.h>

#include <snap/hydro/hydro.hpp>
#include <snap/mesh/meshblock.hpp>

#include "forcing.hpp"

namespace snap {

BodyHeatOptions BodyHeatOptionsImpl::from_yaml(YAML::Node const& forcing) {
  if (!forcing["body-heat"]) return nullptr;

  auto node = forcing["body-heat"];
  auto op = BodyHeatOptionsImpl::create();

  op->dTdt() = node["dTdt"].as<double>(0.0);
  op->pmin() = node["pmin"].as<double>(0.0);
  op->pmax() = node["pmax"].as<double>(1.0);

  TORCH_CHECK(op->pmin() <= op->pmax(),
              "BodyHeatOptions: pmin must not exceed pmax.");

  return op;
}

BodyHeatImpl::BodyHeatImpl(BodyHeatOptions const& options_,
                           torch::nn::Module* p)
    : options(options_) {
  phydro = dynamic_cast<HydroImpl const*>(p);
  reset();
}

void BodyHeatImpl::reset() {
  TORCH_CHECK(phydro, "[BodyHeat] Parent Hydro is null");
}

torch::Tensor BodyHeatImpl::forward(torch::Tensor du, torch::Tensor w,
                                    torch::Tensor temp, double dt) {
  auto interior =
      phydro->pmb->part({0, 0, 0}, PartOptions().exterior(false).ndim(3));
  auto pres = w[IPR].index(interior);
  auto rho = w[IDN].index(interior);
  auto cv = phydro->peos->specific_heat_cv(w, temp).index(interior);
  auto mask =
      torch::logical_and(pres >= options->pmin(), pres <= options->pmax());
  du[IPR].index(interior) += torch::where(mask, dt * options->dTdt() * rho * cv,
                                          torch::zeros_like(pres));
  return du;
}

}  // namespace snap
