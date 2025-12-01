// yaml
#include <yaml-cpp/yaml.h>

// snap
#include "implicit.hpp"

namespace snap {

ImplicitOptions ImplicitOptionsImpl::from_yaml(const std::string& filename,
                                               bool /*verbose*/) {
  auto config = YAML::LoadFile(filename);
  if (!config["integration"]) return nullptr;
  if (!config["integration"]["implicit-scheme"]) return nullptr;
  return from_yaml(config["integration"]["implicit-scheme"]);
}

ImplicitOptions ImplicitOptionsImpl::from_yaml(const YAML::Node& node) {
  auto op = ImplicitOptionsImpl::create();
  op->scheme(node.as<int>());

  switch (op->scheme()) {
    case 0:
      op->type("none");
      break;
    case 1:
      op->type("vic-partial");
      break;
    case 9:
      op->type("vic-full");
      break;
    default:
      TORCH_CHECK(false, "Unsupported implicit scheme");
  }

  return op;
}

}  // namespace snap
