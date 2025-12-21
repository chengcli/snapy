// yaml
#include <yaml-cpp/yaml.h>

// snap
#include <snap/forcing/forcing.hpp>
#include <snap/utils/log.hpp>

#include "hydro.hpp"

namespace snap {

void HydroOptionsImpl::register_forcings_options(std::string const& filename,
                                                 bool verbose) {
  auto config = YAML::LoadFile(filename);
  auto forcing = config["forcing"];
  if (!forcing) return;

  grav() = ConstGravityOptionsImpl::from_yaml(forcing);
  if (grav()) {
    SINFO(HydroOptions) << "gravity options:";
    grav()->report(SINFO());
  }

  coriolis() = CoriolisOptionsImpl::from_yaml(forcing);
  if (coriolis()) {
    coriolis()->coord() = coord();
    SINFO(HydroOptions) << "coriolis options:";
    coriolis()->report(SINFO());
  }

  visc() = DiffusionOptionsImpl::from_yaml(forcing);
  if (visc()) {
    SINFO(HydroOptions) << "diffusion options:";
    visc()->report(SINFO());
  }

  fricHeat() = FricHeatOptionsImpl::from_yaml(forcing);
  if (fricHeat()) {
    SINFO(HydroOptions) << "frictional heating options:";
    fricHeat()->report(SINFO());
  }

  bodyHeat() = BodyHeatOptionsImpl::from_yaml(forcing);
  if (bodyHeat()) {
    bodyHeat()->thermo() = eos()->thermo();
    SINFO(HydroOptions) << "body heating options:";
    bodyHeat()->report(SINFO());
  }

  topCool() = TopCoolOptionsImpl::from_yaml(forcing);
  if (topCool()) {
    topCool()->coord() = coord();
    SINFO(HydroOptions) << "top cooling options:";
    topCool()->report(SINFO());
  }

  botHeat() = BotHeatOptionsImpl::from_yaml(forcing);
  if (botHeat()) {
    botHeat()->coord() = coord();
    SINFO(HydroOptions) << "bottom heating options:";
    botHeat()->report(SINFO());
  }

  relaxBotComp() = RelaxBotCompOptionsImpl::from_yaml(forcing);
  if (relaxBotComp()) {
    SINFO(HydroOptions) << "bottom composition relaxation options:";
    relaxBotComp()->report(SINFO());
  }

  relaxBotTemp() = RelaxBotTempOptionsImpl::from_yaml(forcing);
  if (relaxBotTemp()) {
    SINFO(HydroOptions) << "bottom temperature relaxation options:";
    relaxBotTemp()->report(SINFO());
  }

  relaxBotVelo() = RelaxBotVeloOptionsImpl::from_yaml(forcing);
  if (relaxBotVelo()) {
    SINFO(HydroOptions) << "bottom velocity relaxation options:";
    relaxBotVelo()->report(SINFO());
  }

  topSpongeLyr() = TopSpongeLyrOptionsImpl::from_yaml(forcing);
  if (topSpongeLyr()) {
    topSpongeLyr()->coord() = coord();
    SINFO(HydroOptions) << "top sponge layer options:";
    topSpongeLyr()->report(SINFO());
  }

  botSpongeLyr() = BotSpongeLyrOptionsImpl::from_yaml(forcing);
  if (botSpongeLyr()) {
    botSpongeLyr()->coord() = coord();
    SINFO(HydroOptions) << "bottom sponge layer options:";
    botSpongeLyr()->report(SINFO());
  }

  if (eos()->type() == "plume-eos") {
    plumeForcing() = PlumeForcingOptionsImpl::from_yaml(forcing);
    if (plumeForcing()) {
      SINFO(HydroOptions) << "plume forcing options:";
      plumeForcing()->report(SINFO());
    }
  }
}

HydroOptions HydroOptionsImpl::from_yaml(std::string const& filename,
                                         bool verbose) {
  auto op = HydroOptionsImpl::create();

  // equation of state
  op->eos() = EquationOfStateOptionsImpl::from_yaml(filename, verbose);
  if (verbose) {
    SINFO(HydroOptions) << "equation of state options:";
    op->eos()->report(SINFO());
  }

  // coordinate system
  op->coord() = CoordinateOptionsImpl::from_yaml(filename);
  if (verbose) {
    SINFO(HydroOptions) << "coordinate options:";
    op->coord()->report(SINFO());
  }
  op->coord()->eos() = op->eos();

  // internal boundaries
  op->ib() = InternalBoundaryOptionsImpl::from_yaml(filename);
  if (op->ib()) {
    op->ib()->coord() = op->coord();
    if (verbose) {
      SINFO(HydroOptions) << "internal boundary options:";
      op->ib()->report(SINFO());
    }
  }

  // forcings
  op->register_forcings_options(filename, verbose);

  // primitive projector
  op->proj() = PrimitiveProjectorOptionsImpl::from_yaml(filename);
  if (op->proj()) {
    op->proj()->coord() = op->coord();
    op->proj()->grav() = op->grav();

    if (verbose) {
      SINFO(HydroOptions) << "primitive projector options:";
      op->proj()->report(SINFO());
    }
  }

  // reconstruction
  op->recon1() = ReconstructOptionsImpl::from_yaml(filename, "vertical");
  op->recon1()->eos() = op->eos();

  if (verbose) {
    SINFO(HydroOptions) << "vertical reconstruction options:";
    op->recon1()->report(SINFO());
  }

  op->recon23() = ReconstructOptionsImpl::from_yaml(filename, "horizontal");
  op->recon23()->eos() = op->eos();

  if (verbose) {
    SINFO(HydroOptions) << "horizontal reconstruction options:";
    op->recon23()->report(SINFO());
  }

  // riemann solver
  op->riemann() = RiemannSolverOptionsImpl::from_yaml(filename, "dynamics");

  if (verbose) {
    SINFO(HydroOptions) << "riemann solver options:";
    op->riemann()->report(SINFO());
  }

  // implicit options
  op->icorr() = ImplicitOptionsImpl::from_yaml(filename);
  if (op->icorr()) {
    op->icorr()->grav() = op->grav();

    if (verbose) {
      SINFO(HydroOptions) << "implicit correction options:";
      op->icorr()->report(SINFO());
    }
  }

  // sedimentation
  op->sed() = SedHydroOptionsImpl::from_yaml(filename);
  if (op->sed()) {
    op->sed()->eos() = op->eos();
    op->sed()->sedvel()->grav() = op->grav();
    if (op->fricHeat()) {
      op->fricHeat()->sedvel() = op->sed()->sedvel();
    }

    if (verbose) {
      SINFO(HydroOptions) << "sedimentation options:";
      op->sed()->report(SINFO());
    }
  }

  auto config = YAML::LoadFile(filename);
  auto dyn = config["dynamics"];
  op->verbose() = dyn["verbose"].as<bool>(verbose);
  op->disable_flux_x1() = dyn["disable_flux_x1"].as<bool>(false);
  op->disable_flux_x2() = dyn["disable_flux_x2"].as<bool>(false);
  op->disable_flux_x3() = dyn["disable_flux_x3"].as<bool>(false);

  return op;
}

}  // namespace snap
