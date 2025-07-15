// yaml
#include <yaml-cpp/yaml.h>

// snap
#include "hydro.hpp"

namespace snap {

HydroOptions HydroOptions::from_yaml(std::string const& filename) {
  HydroOptions op;

  op.thermo() = kintera::ThermoOptions::from_yaml(filename);

  auto config = YAML::LoadFile(filename);
  if (config["geometry"]) {
    printf("- reading geometry options from config\n");
    op.coord() = CoordinateOptions::from_yaml(config["geometry"]);
  } else {
    TORCH_WARN("no geometry specified, using default coordinate model");
  }

  // project primitive variables
  printf("- reading projector options from config\n");
  op.proj() = PrimitiveProjectorOptions::from_yaml(config);

  if (!config["dynamics"]) {
    TORCH_WARN("no dynamics specified, using default hydro model");
    return op;
  }

  auto dyn = config["dynamics"];

  // equation of state
  if (dyn["equation-of-state"]) {
    printf("- reading equation of state options from dynamics\n");
    op.eos() = EquationOfStateOptions::from_yaml(dyn["equation-of-state"]);
  } else {
    TORCH_WARN("no equation of state specified, using default EOS model");
  }

  op.eos().coord() = op.coord();
  op.eos().thermo() = op.thermo();

  // reconstruction
  if (dyn["reconstruct"]) {
    printf("- reading reconstruction options from dynamics\n");
    op.recon1() = ReconstructOptions::from_yaml(dyn["reconstruct"], "vertical");
    op.recon23() =
        ReconstructOptions::from_yaml(dyn["reconstruct"], "horizontal");
  } else {
    TORCH_WARN("no reconstruction specified, using default recon model");
  }

  // riemann solver
  if (dyn["riemann-solver"]) {
    printf("- reading riemann solver options from dynamics\n");
    op.riemann() = RiemannSolverOptions::from_yaml(dyn["riemann-solver"]);
  } else {
    TORCH_WARN("no riemann solver specified, using default riemann model");
  }

  op.riemann().eos() = op.eos();

  // internal boundaries
  printf("- reading boundary options from config\n");
  op.ib() = InternalBoundaryOptions::from_yaml(config);

  // implicit options
  op.vic() = ImplicitOptions::from_yaml(config);
  op.vic().recon() = op.recon1();
  op.vic().coord() = op.coord();

  // sedimentation
  if (config["sedimentation"]) {
    printf("- reading sedimentation options from config\n");
    op.sedhydro().sedvel() = SedVelOptions::from_yaml(config);
    op.sedhydro().eos() = op.eos();

    TORCH_CHECK(
        op.sedhydro().sedvel().radius().size() ==
            op.thermo().cloud_ids().size(),
        "Sedimentation radius size must match the number of cloud species.");

    TORCH_CHECK(
        op.sedhydro().sedvel().density().size() ==
            op.thermo().cloud_ids().size(),
        "Sedimentation density size must match the number of cloud species.");
  } else {
    TORCH_WARN("no sedimentation specified");
  }

  // forcings
  if (config["forcing"]) {
    auto forcing = config["forcing"];
    if (forcing["const-gravity"]) {
      printf("- reading constant gravity options from forcing\n");
      op.grav() = ConstGravityOptions::from_yaml(forcing["const-gravity"]);
    } else {
      TORCH_WARN("no constant gravity specified, using default model");
    }

    if (forcing["coriolis"]) {
      printf("- reading coriolis options from forcing\n");
      op.coriolis() = CoriolisOptions::from_yaml(forcing["coriolis"]);
    } else {
      TORCH_WARN("no coriolis specified, using default model");
    }

    if (forcing["diffusion"]) {
      printf("- reading diffusion options from forcing\n");
      op.visc() = DiffusionOptions::from_yaml(forcing["diffusion"]);
    } else {
      TORCH_WARN("no diffusion specified, using default model");
    }

    if (forcing["fric-heat"]) {
      printf("- reading frictional heating options from forcing\n");
      op.fricHeat() = FricHeatOptions::from_yaml(config);
    } else {
      TORCH_WARN("no frictional heating specified, using default model");
    }

    if (forcing["body-heat"]) {
      printf("- reading body heating options from forcing\n");
      op.bodyHeat() = BodyHeatOptions::from_yaml(forcing["body-heat"]);
    } else {
      TORCH_WARN("no body heating specified, using default model");
    }

    if (forcing["top-cool"]) {
      printf("- reading top cooling options from forcing\n");
      op.topCool() = TopCoolOptions::from_yaml(forcing["top-cool"]);
    } else {
      TORCH_WARN("no top cooling specified, using default model");
    }

    if (forcing["bot-heat"]) {
      printf("- reading bottom heating options from forcing\n");
      op.botHeat() = BotHeatOptions::from_yaml(forcing["bot-heat"]);
    } else {
      TORCH_WARN("no bottom heating specified, using default model");
    }

    if (forcing["relax-bot-comp"]) {
      printf("- reading bottom composition relaxation options from forcing\n");
      op.relaxBotComp() =
          RelaxBotCompOptions::from_yaml(forcing["relax-bot-comp"]);
    } else {
      TORCH_WARN(
          "no bottom composition relaxation specified, using default model");
    }

    if (forcing["relax-bot-temp"]) {
      printf("- reading bottom temperature relaxation options from forcing\n");
      op.relaxBotTemp() =
          RelaxBotTempOptions::from_yaml(forcing["relax-bot-temp"]);
    } else {
      TORCH_WARN(
          "no bottom temperature relaxation specified, using default model");
    }

    if (forcing["relax-bot-velo"]) {
      printf("- reading bottom velocity relaxation options from forcing\n");
      op.relaxBotVelo() =
          RelaxBotVeloOptions::from_yaml(forcing["relax-bot-velo"]);
    } else {
      TORCH_WARN(
          "no bottom velocity relaxation specified, using default model");
    }

    if (forcing["top-sponge-lyr"]) {
      printf("- reading top sponge layer options from forcing\n");
      op.topSpongeLyr() =
          TopSpongeLyrOptions::from_yaml(forcing["top-sponge-lyr"]);
    } else {
      TORCH_WARN("no top sponge layer specified, using default model");
    }

    if (forcing["bot-sponge-lyr"]) {
      printf("- reading bottom sponge layer options from forcing\n");
      op.botSpongeLyr() =
          BotSpongeLyrOptions::from_yaml(forcing["bot-sponge-lyr"]);
    } else {
      TORCH_WARN("no bottom sponge layer specified, using default model");
    }
  }

  return op;
}

}  // namespace snap
