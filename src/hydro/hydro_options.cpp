// yaml
#include <yaml-cpp/yaml.h>

// snap
#include <snap/forcing/forcing.hpp>

#include "hydro.hpp"

namespace snap {

void HydroOptionsImpl::register_forcings_options(std::string const& filename) {
  auto config = YAML::LoadFile(filename);
  auto forcing = config["forcing"];
  if (!forcing) return;

  grav() = ConstGravityOptionsImpl::from_yaml(forcing);

  coriolis() = CoriolisOptionsImpl::from_yaml(forcing);
  if (coriolis()) {
    coriolis()->coord() = coord();
  }

  visc() = DiffusionOptionsImpl::from_yaml(forcing);

  fricHeat() = FricHeatOptionsImpl::from_yaml(forcing);

  bodyHeat() = BodyHeatOptionsImpl::from_yaml(forcing);
  if (bodyHeat()) {
    bodyHeat()->thermo() = eos()->thermo();
  }

  topCool() = TopCoolOptionsImpl::from_yaml(forcing);
  if (topCool()) {
    topCool()->coord() = coord();
  }

  botHeat() = BotHeatOptionsImpl::from_yaml(forcing);
  if (botHeat()) {
    botHeat()->coord() = coord();
  }

  relaxBotComp() = RelaxBotCompOptionsImpl::from_yaml(forcing);

  relaxBotTemp() = RelaxBotTempOptionsImpl::from_yaml(forcing);

  relaxBotVelo() = RelaxBotVeloOptionsImpl::from_yaml(forcing);

  topSpongeLyr() = TopSpongeLyrOptionsImpl::from_yaml(forcing);
  if (topSpongeLyr()) {
    topSpongeLyr()->coord() = coord();
  }

  botSpongeLyr() = BotSpongeLyrOptionsImpl::from_yaml(forcing);
  if (botSpongeLyr()) {
    botSpongeLyr()->coord() = coord();
  }

  if (eos()->type() == "plume-eos") {
    plumeForcing() = PlumeForcingOptionsImpl::from_yaml(forcing);
  }
}

HydroOptions HydroOptionsImpl::from_yaml(std::string const& filename) {
  auto op = HydroOptionsImpl::create();

  // internal boundaries
  op->ib() = InternalBoundaryOptionsImpl::from_yaml(filename);

  // coordinate system
  op->coord() = CoordinateOptionsImpl::from_yaml(filename);

  // equation of state
  op->eos() = EquationOfStateOptionsImpl::from_yaml(filename);

  // link eos and coord
  op->eos()->coord() = op->coord();
  op->coord()->eos() = op->eos();

  // forcings
  op->register_forcings_options(filename);

  // primitive projector
  op->proj() = PrimitiveProjectorOptionsImpl::from_yaml(filename);
  if (op->proj()) {
    op->proj()->coord() = op->coord();
    op->proj()->grav() = op->grav();
  }

  // reconstruction
  op->recon1() = ReconstructOptionsImpl::from_yaml(filename, "vertical");
  if (op->recon1()) {
    op->recon1()->eos() = op->eos();
  }

  op->recon23() = ReconstructOptionsImpl::from_yaml(filename, "horizontal");
  if (op->recon23()) {
    op->recon23()->eos() = op->eos();
  }

  // riemann solver
  op->riemann() = RiemannSolverOptionsImpl::from_yaml(filename);
  if (op->riemann()) {
    op->riemann()->eos() = op->eos();
  }

  // implicit options
  op->icorr() = ImplicitOptionsImpl::from_yaml(filename);
  if (op->icorr()) {
    op->icorr()->coord() = op->coord();
    op->icorr()->grav() = op->grav();
  }

  // sedimentation
  op->sed() = SedHydroOptionsImpl::from_yaml(filename);
  if (op->sed()) {
    op->sed()->eos() = op->eos();
    op->sed()->sedvel()->grav() = op->grav();
    op->fricHeat()->sedvel() = op->sed()->sedvel();
  }

  auto config = YAML::LoadFile(filename);
  auto dyn = config["dynamics"];
  op->verbose() = dyn["verbose"].as<bool>(false);
  op->disable_flux_x1() = dyn["disable_flux_x1"].as<bool>(false);
  op->disable_flux_x2() = dyn["disable_flux_x2"].as<bool>(false);
  op->disable_flux_x3() = dyn["disable_flux_x3"].as<bool>(false);

  return op;
}

}  // namespace snap
