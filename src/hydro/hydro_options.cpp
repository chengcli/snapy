// C/C++
#include <algorithm>

// yaml
#include <yaml-cpp/yaml.h>

// snap
#include <snap/snap.h>

#include "hydro.hpp"

namespace snap {

HydroOptions HydroOptions::from_yaml(std::string const& filename,
                                     DistributeInfo dist) {
  HydroOptions op;

  op.thermo() = kintera::ThermoOptions::from_yaml(filename);

  TORCH_CHECK(
      NMASS == 0 ||
          op.thermo().vapor_ids().size() + op.thermo().cloud_ids().size() ==
              1 + NMASS,
      "Athena++ style indexing is enabled (NMASS > 0), but the number of "
      "vapor and cloud species in the thermodynamics options does not match "
      "the expected number of vapor + cloud species = ",
      1 + NMASS);

  auto config = YAML::LoadFile(filename);
  if (config["geometry"]) {
    op.coord() = CoordinateOptions::from_yaml(config["geometry"], dist);
  }

  // project primitive variables
  op.proj() = PrimitiveProjectorOptions::from_yaml(config);

  if (!config["dynamics"]) return op;

  auto dyn = config["dynamics"];

  op.disable_dynamics() = dyn["disable"].as<bool>(false);

  // equation of state
  if (dyn["equation-of-state"]) {
    op.eos() = EquationOfStateOptions::from_yaml(dyn["equation-of-state"]);
    op.coord().eos_type() = op.eos().type();
  }

  op.eos().coord() = op.coord();
  op.eos().thermo() = op.thermo();

  // reconstruction
  if (dyn["reconstruct"]) {
    op.recon1() = ReconstructOptions::from_yaml(dyn, "vertical");
    op.recon23() = ReconstructOptions::from_yaml(dyn, "horizontal");
  }

  // riemann solver
  if (dyn["riemann-solver"]) {
    op.riemann() = RiemannSolverOptions::from_yaml(dyn["riemann-solver"]);
  }

  op.riemann().eos() = op.eos();

  // internal boundaries
  op.ib() = InternalBoundaryOptions::from_yaml(config);

  // implicit options
  op.imp() = ImplicitOptions::from_yaml(config);
  op.imp().coord() = op.coord();

  // sedimentation
  if (config["sedimentation"]) {
    op.sed().sedvel() = SedVelOptions::from_yaml(config);
    op.sed().eos() = op.eos();

    // check all precipitating particles are in the clouds
    std::unordered_set<int> cloud_set(op.thermo().cloud_ids().begin(),
                                      op.thermo().cloud_ids().end());
    auto particle_ids = op.sed().sedvel().particle_ids();
    auto pass = std::all_of(particle_ids.begin(), particle_ids.end(),
                            [&](int x) { return cloud_set.count(x); });

    TORCH_CHECK(pass, "Missing sedimentation particles in the clouds.");

    // setup hydro ids
    auto hydro_species = op.thermo().species();
    for (auto const& p : op.sed().sedvel().species()) {
      auto it = std::find(hydro_species.begin(), hydro_species.end(), p);
      op.sed().hydro_ids().push_back(Index::ICY - 1 + it -
                                     hydro_species.begin());
    }
  }

  // forcings
  if (config["forcing"]) {
    auto forcing = config["forcing"];
    if (forcing["const-gravity"]) {
      op.grav() = ConstGravityOptions::from_yaml(forcing["const-gravity"]);
    }

    if (forcing["coriolis"]) {
      op.coriolis() = CoriolisOptions::from_yaml(forcing["coriolis"]);
    }

    if (forcing["diffusion"]) {
      op.visc() = DiffusionOptions::from_yaml(forcing["diffusion"]);
    }

    if (forcing["fric-heat"]) {
      op.fricHeat() = FricHeatOptions::from_yaml(config);
    }

    if (forcing["body-heat"]) {
      op.bodyHeat() = BodyHeatOptions::from_yaml(forcing["body-heat"]);
    }

    if (forcing["top-cool"]) {
      op.topCool() = TopCoolOptions::from_yaml(forcing["top-cool"]);
    }

    if (forcing["bot-heat"]) {
      op.botHeat() = BotHeatOptions::from_yaml(forcing["bot-heat"]);
    }

    if (forcing["relax-bot-comp"]) {
      op.relaxBotComp() =
          RelaxBotCompOptions::from_yaml(forcing["relax-bot-comp"]);
    }

    if (forcing["relax-bot-temp"]) {
      op.relaxBotTemp() =
          RelaxBotTempOptions::from_yaml(forcing["relax-bot-temp"]);
    }

    if (forcing["relax-bot-velo"]) {
      op.relaxBotVelo() =
          RelaxBotVeloOptions::from_yaml(forcing["relax-bot-velo"]);
    }

    if (forcing["top-sponge-lyr"]) {
      op.topSpongeLyr() =
          TopSpongeLyrOptions::from_yaml(forcing["top-sponge-lyr"]);
    }

    if (forcing["bot-sponge-lyr"]) {
      op.botSpongeLyr() =
          BotSpongeLyrOptions::from_yaml(forcing["bot-sponge-lyr"]);
    }
  }

  return op;
}

}  // namespace snap
