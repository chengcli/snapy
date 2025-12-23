// yaml
#include <yaml-cpp/yaml.h>

// snap
#include <snap/hydro/hydro.hpp>

#include "forcing.hpp"

namespace snap {

BodyHeatOptions BodyHeatOptionsImpl::from_yaml(YAML::Node const& forcing) {
  if (!forcing["body-heat"]) return nullptr;

  auto node = forcing["body-heat"];
  auto op = BodyHeatOptionsImpl::create();

  op->dTdt() = node["dTdt"].as<double>(0.0);
  op->pmin() = node["pmin"].as<double>(0.0);
  op->pmax() = node["pmax"].as<double>(1.0);

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
  // auto wtop = w.select(-1, ie);
  // auto ivol = pthermo->compute("DY->V", {wtop[IDN], wtop.narrow(0, 1, ny)});
  // auto cv = pthermo->compute("VT->cv", {ivol, temp});
  return du;
}

}  // namespace snap
