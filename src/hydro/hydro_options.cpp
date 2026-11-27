// C/C++
#include <cstdlib>
#include <string>

// yaml
#include <yaml-cpp/yaml.h>

// snap
#include <snap/forcing/forcing.hpp>
#include <snap/input/check_keys.hpp>
#include <snap/input/report.hpp>
#include <snap/utils/log.hpp>

#include "hydro.hpp"

namespace snap {

#include <snap_options/hydro_read.inc>

HydroOptions HydroOptionsImpl::from_yaml(std::string const& filename,
                                         bool verbose) {
  auto op = HydroOptionsImpl::create();

  // ------------ equation of state ------------ //
  op->eos() = EquationOfStateOptionsImpl::from_yaml(filename, verbose);

  // ------------- reconstruction ------------ //
  op->recon1() = ReconstructOptionsImpl::from_yaml(filename, "vertical");

  op->recon23() = ReconstructOptionsImpl::from_yaml(filename, "horizontal");

  // ------------ riemann solver ------------ //
  op->riemann() = RiemannSolverOptionsImpl::from_yaml(filename, "dynamics");

  // ---------- implicit correction --------- //
  op->icorr() = ImplicitOptionsImpl::from_yaml(filename);

  // ------------ sedimentation ------------- //
  op->sed() = SedHydroOptionsImpl::from_yaml(filename);

  // -------------- others ------------------ //
  auto config = YAML::LoadFile(filename);
  // dynamics/* rows and the SNAP_ANOMALY override (src/options/hydro.yaml)
  read_hydro_rows(*op, config["dynamics"], "dynamics", verbose);
  if (verbose)
    report_input(SINFO(HydroOptions), "dynamics", config["dynamics"]);
  if (verbose && op->anomaly()) {
    SINFO(HydroOptions) << "anomaly storage of hydro_u {IDN, IPR} is on\n";
  }

  // --------------- forcings --------------- //
  auto forcing = config["forcing"];
  if (!forcing) return op;

  TORCH_CHECK(forcing.IsMap() || forcing.IsNull(),
              "HydroOptions: forcing must be a map.");
  if (forcing.IsNull()) return op;
  TORCH_CHECK(!forcing["fric-heat"].IsDefined(),
              "HydroOptions: 'forcing/fric-heat' has been removed. The x1 "
              "face gravity work already carries the sedimentation channel "
              "of the mass flux, so this key would apply the "
              "precipitation potential-energy release twice. Delete it.");
  check_keys(forcing, "forcing",
             {"const-gravity", "coriolis", "diffusion", "body-heat", "top-cool",
              "bot-heat", "relax-bot-comp", "relax-bot-temp", "relax-bot-velo",
              "top-sponge-lyr", "bot-sponge-lyr", "plume-forcing"});

  op->grav() = ConstGravityOptionsImpl::from_yaml(forcing);
  if (op->grav()) {
    if (op->disable_flux_x1()) op->grav()->grav1(0.);
    if (op->disable_flux_x2()) op->grav()->grav2(0.);
    if (op->disable_flux_x3()) op->grav()->grav3(0.);
  }

  op->coriolis() = CoriolisOptionsImpl::from_yaml(forcing);
  op->diffusion() = DiffusionOptionsImpl::from_yaml(forcing);
  op->bodyHeat() = BodyHeatOptionsImpl::from_yaml(forcing);
  op->topCool() = TopCoolOptionsImpl::from_yaml(forcing);
  op->botHeat() = BotHeatOptionsImpl::from_yaml(forcing);
  op->relaxBotComp() = RelaxBotCompOptionsImpl::from_yaml(forcing);
  op->relaxBotTemp() = RelaxBotTempOptionsImpl::from_yaml(forcing);
  op->relaxBotVelo() = RelaxBotVeloOptionsImpl::from_yaml(forcing);
  op->topSpongeLyr() = TopSpongeLyrOptionsImpl::from_yaml(forcing);
  op->botSpongeLyr() = BotSpongeLyrOptionsImpl::from_yaml(forcing);

  if (op->eos()->type() == "plume-eos") {
    op->plumeForcing() = PlumeForcingOptionsImpl::from_yaml(forcing);
  }

  report_input(SINFO(HydroOptions), "forcing", forcing);
  return op;
}

HydroOptions HydroOptionsImpl::clone() const {
  auto op = HydroOptionsImpl::create();

  op->verbose() = verbose();
  op->disable_flux_x1() = disable_flux_x1();
  op->disable_flux_x2() = disable_flux_x2();
  op->disable_flux_x3() = disable_flux_x3();
  op->conversion_order() = conversion_order();
  op->anomaly() = anomaly();
  op->flux_covariance() = flux_covariance();
  op->x1_mass_covariance() = x1_mass_covariance();
  op->perturbation_flux() = perturbation_flux();
  op->precision() = precision();
  op->conserve_axial_momentum() = conserve_axial_momentum();
  if (grav()) op->grav() = grav()->clone();
  if (coriolis()) op->coriolis() = coriolis()->clone();
  if (diffusion()) op->diffusion() = diffusion()->clone();
  if (bodyHeat()) op->bodyHeat() = bodyHeat()->clone();
  if (topCool()) op->topCool() = topCool()->clone();
  if (botHeat()) op->botHeat() = botHeat()->clone();
  if (relaxBotComp()) op->relaxBotComp() = relaxBotComp()->clone();
  if (relaxBotTemp()) op->relaxBotTemp() = relaxBotTemp()->clone();
  if (relaxBotVelo()) op->relaxBotVelo() = relaxBotVelo()->clone();
  if (topSpongeLyr()) op->topSpongeLyr() = topSpongeLyr()->clone();
  if (botSpongeLyr()) op->botSpongeLyr() = botSpongeLyr()->clone();
  if (plumeForcing()) op->plumeForcing() = plumeForcing()->clone();

  // TODO(cli)
  /*if (eos()) op->eos() = eos()->clone();
  if (recon1()) op->recon1() = recon1()->clone();
  if (recon23()) op->recon23() = recon23()->clone();
  if (riemann()) op->riemann() = riemann()->clone();
  if (icorr()) op->icorr() = icorr()->clone();
  if (sed()) op->sed() = sed()->clone();*/

  return op;
}

}  // namespace snap
