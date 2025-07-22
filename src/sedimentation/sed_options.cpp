// yaml
#include <yaml-cpp/yaml.h>

// kintera
#include <kintera/species.hpp>

// snap
#include "sedimentation.hpp"

namespace snap {

SedVelOptions SedVelOptions::from_yaml(YAML::Node const& root) {
  SedVelOptions op;

  if (!root["forcing"]) return op;
  if (!root["forcing"]["const-gravity"]) return op;

  op.grav() = root["forcing"]["const-gravity"]["grav1"].as<double>(0.0);

  if (!root["sedimentation"]) return op;

  auto config = root["sedimentation"];

  // get all sedimentation particles
  std::set<std::string> particle_names;
  for (auto r : config["radius"]) {
    auto name = r.first.as<std::string>();
    particle_names.insert(name);
  }

  for (auto r : config["density"]) {
    auto name = r.first.as<std::string>();
    particle_names.insert(name);
  }

  for (auto r : config["const-vsed"]) {
    auto name = r.first.as<std::string>();
    particle_names.insert(name);
  }

  // get particle ids
  for (auto& name : particle_names) {
    auto it = std::find(kintera::species_names.begin(),
                        kintera::species_names.end(), name);
    if (it == kintera::species_names.end()) {
      TORCH_CHECK(false, "Sedimentation particle '", name,
                  "' is not a valid species.");
    }
    int id = it - kintera::species_names.begin();
    op.particle_ids().push_back(id);
  }

  op.radius().resize(op.particle_ids().size(), 0.);
  op.density().resize(op.particle_ids().size(), 0.);
  op.const_vsed().resize(op.particle_ids().size(), 0.);

  // read particle radius
  auto species = op.species();
  for (auto r : config["radius"]) {
    auto name = r.first.as<std::string>();
    auto it = std::find(species.begin(), species.end(), name);
    auto radius = r.second.as<double>();
    TORCH_CHECK(radius > 0., "Sedimentation radius must be positive.");
    op.radius()[it - species.begin()] = radius;
  }

  // read particle density
  for (auto r : config["density"]) {
    auto name = r.first.as<std::string>();
    auto it = std::find(species.begin(), species.end(), name);
    auto density = r.second.as<double>();
    TORCH_CHECK(density > 0., "Sedimentation density must be positive.");
    op.density()[it - species.begin()] = density;
  }

  // read particle constant sedimentation velocity
  for (auto r : config["const-vsed"]) {
    auto name = r.first.as<std::string>();
    auto it = std::find(species.begin(), species.end(), name);
    op.const_vsed()[it - species.begin()] = r.second.as<double>();
  }

  op.a_diameter() = config["a-diameter"].as<double>(2.827e-10);
  op.a_epsilon_LJ() = config["a-epsilon-LJ"].as<double>(59.7e-7);
  op.a_mass() = config["a-mass"].as<double>(3.34e-27);
  op.upper_limit() = config["upper-limit"].as<double>(5.e3);

  return op;
}

std::vector<std::string> SedVelOptions::species() const {
  std::vector<std::string> species_list;

  for (int i = 0; i < particle_ids().size(); ++i) {
    species_list.push_back(kintera::species_names[particle_ids()[i]]);
  }

  return species_list;
}

}  // namespace snap
