// yaml
#include <yaml-cpp/yaml.h>

// snap
#include <snap/forcing/forcing.hpp>
#include <snap/utils/log.hpp>

#include "hydro.hpp"

namespace snap {

HydroOptions HydroOptionsImpl::from_yaml(std::string const& filename,
                                         bool verbose) {
  auto op = HydroOptionsImpl::create();

  // ------------ equation of state ------------ //
  op->eos() = EquationOfStateOptionsImpl::from_yaml(filename, verbose);
  if (verbose) op->eos()->report(SINFO(HydroOptions));

  // ---------- primitive projector ----------- //
  op->proj() = PrimitiveProjectorOptionsImpl::from_yaml(filename);
  if (op->proj() && verbose) op->proj()->report(SINFO(HydroOptions));

  // ------------- reconstruction ------------ //
  op->recon1() = ReconstructOptionsImpl::from_yaml(filename, "vertical");
  if (verbose) op->recon1()->report(SINFO(HydroOptions : vertical));

  op->recon23() = ReconstructOptionsImpl::from_yaml(filename, "horizontal");
  if (verbose) op->recon23()->report(SINFO(HydroOptions : horizontal));

  // ------------ riemann solver ------------ //
  op->riemann() = RiemannSolverOptionsImpl::from_yaml(filename, "dynamics");
  if (verbose) op->riemann()->report(SINFO(HydroOptions));

  // ---------- implicit correction --------- //
  op->icorr() = ImplicitOptionsImpl::from_yaml(filename);
  if (op->icorr() && verbose) op->icorr()->report(SINFO(HydroOptions));

  // ------------ sedimentation ------------- //
  op->sed() = SedHydroOptionsImpl::from_yaml(filename);
  if (op->sed() && verbose) op->sed()->report(SINFO(HydroOptions));

  // -------------- others ------------------ //
  auto config = YAML::LoadFile(filename);
  auto dyn = config["dynamics"];
  if (dyn) {
    op->verbose() = dyn["verbose"].as<bool>(verbose);
    op->disable_flux_x1() = dyn["disable-flux-x1"].as<bool>(false);
    op->disable_flux_x2() = dyn["disable-flux-x2"].as<bool>(false);
    op->disable_flux_x3() = dyn["disable-flux-x3"].as<bool>(false);
  }

  // --------------- forcings --------------- //
  auto forcing = config["forcing"];
  if (!forcing) return op;

  op->grav() = ConstGravityOptionsImpl::from_yaml(forcing);
  if (op->grav()) {
    if (op->disable_flux_x1()) op->grav()->grav1(0.);
    if (op->disable_flux_x2()) op->grav()->grav2(0.);
    if (op->disable_flux_x3()) op->grav()->grav3(0.);
    op->grav()->report(SINFO(HydroOptions));
  }

  op->coriolis() = CoriolisOptionsImpl::from_yaml(forcing);
  if (op->coriolis()) op->coriolis()->report(SINFO(HydroOptions));

  op->visc() = DiffusionOptionsImpl::from_yaml(forcing);
  if (op->visc()) op->visc()->report(SINFO(HydroOptions));

  op->fricHeat() = FricHeatOptionsImpl::from_yaml(forcing);
  if (op->fricHeat()) op->fricHeat()->report(SINFO(HydroOptions));

  op->bodyHeat() = BodyHeatOptionsImpl::from_yaml(forcing);
  if (op->bodyHeat()) op->bodyHeat()->report(SINFO(HydroOptions));

  op->topCool() = TopCoolOptionsImpl::from_yaml(forcing);
  if (op->topCool()) op->topCool()->report(SINFO(HydroOptions));

  op->botHeat() = BotHeatOptionsImpl::from_yaml(forcing);
  if (op->botHeat()) op->botHeat()->report(SINFO(HydroOptions));

  op->relaxBotComp() = RelaxBotCompOptionsImpl::from_yaml(forcing);
  if (op->relaxBotComp()) op->relaxBotComp()->report(SINFO(HydroOptions));

  op->relaxBotTemp() = RelaxBotTempOptionsImpl::from_yaml(forcing);
  if (op->relaxBotTemp()) op->relaxBotTemp()->report(SINFO(HydroOptions));

  op->relaxBotVelo() = RelaxBotVeloOptionsImpl::from_yaml(forcing);
  if (op->relaxBotVelo()) op->relaxBotVelo()->report(SINFO(HydroOptions));

  op->topSpongeLyr() = TopSpongeLyrOptionsImpl::from_yaml(forcing);
  if (op->topSpongeLyr()) op->topSpongeLyr()->report(SINFO(HydroOptions));

  op->botSpongeLyr() = BotSpongeLyrOptionsImpl::from_yaml(forcing);
  if (op->botSpongeLyr()) op->botSpongeLyr()->report(SINFO(HydroOptions));

  if (op->eos()->type() == "plume-eos") {
    op->plumeForcing() = PlumeForcingOptionsImpl::from_yaml(forcing);
    if (op->plumeForcing()) op->plumeForcing()->report(SINFO(HydroOptions));
  }

  return op;
}

}  // namespace snap
