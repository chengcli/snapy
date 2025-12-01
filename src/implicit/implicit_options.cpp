// yaml
#include <yaml-cpp/yaml.h>

// snap
#include "implicit.hpp"

namespace snap {

ImplicitOptions ImplicitOptionsImpl::from_yaml(const std::string& filename) {
  auto config = YAML::LoadFile(filename);
  if (!config["implicit-scheme"]) return nullptr;
  return from_yaml(config["implicit-scheme"]);
}

ImplicitOptions ImplicitOptionsImpl::from_yaml(const YAML::Node& node) {
  auto op = ImplicitOptionsImpl::create();

  switch (node.as<int>(0)) {
    case 0:
      op->type() = "none";
      op->scheme() = 0;
      break;
    case 1:
      op->type() = "vic-partial";
      op->scheme() = 1;
      break;
    case 9:
      op->type() = "vic-full";
      op->scheme() = 9;
      break;
    default:
      TORCH_CHECK(false, "Unsupported implicit scheme");
  }

  return op;
}

}  // namespace snap
