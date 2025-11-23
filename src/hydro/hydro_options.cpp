// C/C++
#include <algorithm>

// yaml
#include <yaml-cpp/yaml.h>

// snap
#include <snap/snap.h>

#include <snap/registry.hpp>

#include "hydro.hpp"

namespace snap {

HydroOptions HydroOptions::from_yaml(std::string const& filename,
                                     LayoutOptions layout) {
  HydroOptions op;

  auto config = YAML::LoadFile(filename);
  if (config["geometry"]) {
    op.coord() = CoordinateOptions::from_yaml(config["geometry"], layout);
  }

  // project primitive variables
  op.proj() = PrimitiveProjectorOptions::from_yaml(config);

  if (!config["dynamics"]) return op;

  auto dyn = config["dynamics"];

  op.disable_flux_x1() = dyn["disable_flux_x1"].as<bool>(false);
  op.disable_flux_x2() = dyn["disable_flux_x2"].as<bool>(false);
  op.disable_flux_x3() = dyn["disable_flux_x3"].as<bool>(false);

  // equation of state
  if (dyn["equation-of-state"]) {
    op.eos() =
        EquationOfStateOptions::from_yaml(dyn["equation-of-state"], filename);
    op.coord().eos_type() = op.eos().type();
  }
  op.eos().coord() = op.coord();

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
    std::unordered_set<int> cloud_set(op.eos().thermo().cloud_ids().begin(),
                                      op.eos().thermo().cloud_ids().end());
    auto particle_ids = op.sed().sedvel().particle_ids();
    auto pass = std::all_of(particle_ids.begin(), particle_ids.end(),
                            [&](int x) { return cloud_set.count(x); });

    TORCH_CHECK(pass, "Missing sedimentation particles in the clouds.");

    // setup hydro ids
    auto hydro_species = op.eos().thermo().species();
    for (auto const& p : op.sed().sedvel().species()) {
      auto it = std::find(hydro_species.begin(), hydro_species.end(), p);
      op.sed().hydro_ids().push_back(Index::ICY - 1 + it -
                                     hydro_species.begin());
    }
  }

  // forcings
  if (config["forcing"]) register_forcings_options(op, config, layout);

  return op;
}

}  // namespace snap
