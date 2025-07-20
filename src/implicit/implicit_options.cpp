// yaml
#include <yaml-cpp/yaml.h>

// snap
#include "implicit.hpp"

namespace snap {

ImplicitOptions ImplicitOptions::from_yaml(const YAML::Node& root) {
  ImplicitOptions op;

  if (!root["integration"]) return op;

  switch (root["integration"]["implicit-scheme"].as<int>(0)) {
    case 0:
      op.type() = "none";
      op.scheme() = 0;
      break;
    case 1:
      op.type() = "vic-partial";
      op.scheme() = 1;
      break;
    case 9:
      op.type() = "vic-full";
      op.scheme() = 9;
      break;
    default:
      TORCH_CHECK(false, "Unsupported implicit scheme");
  }

  if (!root["forcing"]) return op;
  if (!root["forcing"]["const-gravity"]) return op;

  op.grav() = root["forcing"]["const-gravity"]["grav1"].as<double>(0.0);

  return op;
}

}  // namespace snap
