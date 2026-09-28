// C/C++
#include <algorithm>

// torch
#include <c10/util/Exception.h>

// snap
#include "check_keys.hpp"

namespace snap {

void check_keys(YAML::Node const& node, std::string const& path,
                std::vector<std::string> const& keys, std::string const& other,
                std::vector<std::string> const& other_keys) {
  if (!node.IsMap()) return;
  auto joined = [](std::vector<std::string> const& list) {
    std::string s;
    for (auto const& k : list) s += (s.empty() ? "" : ", ") + k;
    return s;
  };
  for (auto const& item : node) {
    auto key = item.first.as<std::string>();
    auto listed = [&key](std::vector<std::string> const& list) {
      return std::find(list.begin(), list.end(), key) != list.end();
    };
    TORCH_CHECK(listed(keys) || listed(other_keys), "unknown key '", path, "/",
                key, "'. Valid keys: ", joined(keys),
                other.empty() ? "" : "; read by " + other + ": ",
                joined(other_keys), ".");
  }
}

}  // namespace snap
