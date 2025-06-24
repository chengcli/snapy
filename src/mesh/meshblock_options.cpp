// yaml
#include <yaml-cpp/yaml.h>

// snap
#include "meshblock.hpp"

namespace snap {

MeshBlockOptions MeshBlockOptions::from_yaml(std::string input_file) {
  MeshBlockOptions op;

  op.hydro() = HydroOptions::from_yaml(input_file);
  op.intg() = IntegratorOptions::from_yaml(input_file);

  auto config = YAML::LoadFile(input_file);

  if (!config["dynamics"]) return op;
  if (!config["dynamics"]["boundary-condition"]) return op;
  if (!config["dynamics"]["boundary-condition"]["external"]) return op;

  if (config["dynamics"]["boundary-condition"]["external"]["x1-inner"]) }

}  // namespace snap
