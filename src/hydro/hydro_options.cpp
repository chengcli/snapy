// yaml
#include <yaml-cpp/yaml.h>

// snap
#include <snap/forcing/forcing.hpp>

#include "hydro.hpp"

namespace snap {

void HydroOptionsImpl::register_forcings_options(std::string const& filename,
                                                 bool verbose) {
  auto config = YAML::LoadFile(filename);
  auto forcing = config["forcing"];
  if (!forcing) return;

  grav() = ConstGravityOptionsImpl::from_yaml(forcing);
  if (grav() && verbose) {
    std::cout << "[HydroOptions] gravity options:" << std::endl;
    grav()->report(std::cout);
  }

  coriolis() = CoriolisOptionsImpl::from_yaml(forcing);
  if (coriolis()) {
    coriolis()->coord() = coord();
    if (verbose) {
      std::cout << "[HydroOptions] coriolis options:" << std::endl;
      coriolis()->report(std::cout);
    }
  }

  visc() = DiffusionOptionsImpl::from_yaml(forcing);
  if (visc() && verbose) {
    std::cout << "[HydroOptions] diffusion options:" << std::endl;
    visc()->report(std::cout);
  }

  fricHeat() = FricHeatOptionsImpl::from_yaml(forcing);
  if (fricHeat() && verbose) {
    std::cout << "[HydroOptions] frictional heating options:" << std::endl;
    fricHeat()->report(std::cout);
  }

  bodyHeat() = BodyHeatOptionsImpl::from_yaml(forcing);
  if (bodyHeat()) {
    bodyHeat()->thermo() = eos()->thermo();
    if (verbose) {
      std::cout << "[HydroOptions] body heating options:" << std::endl;
      bodyHeat()->report(std::cout);
    }
  }

  topCool() = TopCoolOptionsImpl::from_yaml(forcing);
  if (topCool()) {
    topCool()->coord() = coord();
    if (verbose) {
      std::cout << "[HydroOptions] top cooling options:" << std::endl;
      topCool()->report(std::cout);
    }
  }

  botHeat() = BotHeatOptionsImpl::from_yaml(forcing);
  if (botHeat()) {
    botHeat()->coord() = coord();
    if (verbose) {
      std::cout << "[HydroOptions] bottom heating options:" << std::endl;
      botHeat()->report(std::cout);
    }
  }

  relaxBotComp() = RelaxBotCompOptionsImpl::from_yaml(forcing);
  if (relaxBotComp() && verbose) {
    std::cout << "[HydroOptions] bottom composition relaxation options:"
              << std::endl;
    relaxBotComp()->report(std::cout);
  }

  relaxBotTemp() = RelaxBotTempOptionsImpl::from_yaml(forcing);
  if (relaxBotTemp() && verbose) {
    std::cout << "[HydroOptions] bottom temperature relaxation options:"
              << std::endl;
    relaxBotTemp()->report(std::cout);
  }

  relaxBotVelo() = RelaxBotVeloOptionsImpl::from_yaml(forcing);
  if (relaxBotVelo() && verbose) {
    std::cout << "[HydroOptions] bottom velocity relaxation options:"
              << std::endl;
    relaxBotVelo()->report(std::cout);
  }

  topSpongeLyr() = TopSpongeLyrOptionsImpl::from_yaml(forcing);
  if (topSpongeLyr()) {
    topSpongeLyr()->coord() = coord();
    if (verbose) {
      std::cout << "[HydroOptions] top sponge layer options:" << std::endl;
      topSpongeLyr()->report(std::cout);
    }
  }

  botSpongeLyr() = BotSpongeLyrOptionsImpl::from_yaml(forcing);
  if (botSpongeLyr()) {
    botSpongeLyr()->coord() = coord();
    if (verbose) {
      std::cout << "[HydroOptions] bottom sponge layer options:" << std::endl;
      botSpongeLyr()->report(std::cout);
    }
  }

  if (eos()->type() == "plume-eos") {
    plumeForcing() = PlumeForcingOptionsImpl::from_yaml(forcing);
    if (plumeForcing() && verbose) {
      std::cout << "[HydroOptions] plume forcing options:" << std::endl;
      plumeForcing()->report(std::cout);
    }
  }
}

HydroOptions HydroOptionsImpl::from_yaml(std::string const& filename,
                                         bool verbose) {
  auto op = HydroOptionsImpl::create();

  // coordinate system
  op->coord() = CoordinateOptionsImpl::from_yaml(filename);
  if (verbose) {
    std::cout << "[HydroOptions] coordinate options:" << std::endl;
    op->coord()->report(std::cout);
  }

  // equation of state
  op->eos() = EquationOfStateOptionsImpl::from_yaml(filename, verbose);
  if (verbose) {
    std::cout << "[HydroOptions] equation of state options:" << std::endl;
    op->eos()->report(std::cout);
  }

  // link eos and coord
  op->eos()->coord() = op->coord();
  op->coord()->eos() = op->eos();

  // internal boundaries
  op->ib() = InternalBoundaryOptionsImpl::from_yaml(filename);
  if (op->ib()) {
    op->ib()->coord() = op->coord();
    if (verbose) {
      std::cout << "[HydroOptions] internal boundary options:" << std::endl;
      op->ib()->report(std::cout);
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
      std::cout << "[HydroOptions] primitive projector options:" << std::endl;
      op->proj()->report(std::cout);
    }
  }

  // reconstruction
  op->recon1() = ReconstructOptionsImpl::from_yaml(filename, "vertical");
  if (op->recon1()) {
    op->recon1()->eos() = op->eos();

    if (verbose) {
      std::cout << "[HydroOptions] vertical reconstruction options:"
                << std::endl;
      op->recon1()->report(std::cout);
    }
  }

  op->recon23() = ReconstructOptionsImpl::from_yaml(filename, "horizontal");
  if (op->recon23()) {
    op->recon23()->eos() = op->eos();

    if (verbose) {
      std::cout << "[HydroOptions] horizontal reconstruction options:"
                << std::endl;
      op->recon23()->report(std::cout);
    }
  }

  // riemann solver
  op->riemann() = RiemannSolverOptionsImpl::from_yaml(filename, "dynamics");
  if (op->riemann()) {
    op->riemann()->eos() = op->eos();

    if (verbose) {
      std::cout << "[HydroOptions] riemann solver options:" << std::endl;
      op->riemann()->report(std::cout);
    }
  }

  // implicit options
  op->icorr() = ImplicitOptionsImpl::from_yaml(filename);
  if (op->icorr()) {
    op->icorr()->coord() = op->coord();
    op->icorr()->grav() = op->grav();

    if (verbose) {
      std::cout << "[HydroOptions] implicit correction options:" << std::endl;
      op->icorr()->report(std::cout);
    }
  }

  // sedimentation
  op->sed() = SedHydroOptionsImpl::from_yaml(filename);
  if (op->sed()) {
    op->sed()->eos() = op->eos();
    op->sed()->sedvel()->grav() = op->grav();
    op->fricHeat()->sedvel() = op->sed()->sedvel();

    if (verbose) {
      std::cout << "[HydroOptions] sedimentation options:" << std::endl;
      op->sed()->report(std::cout);
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
