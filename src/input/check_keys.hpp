#pragma once

// C/C++
#include <string>
#include <vector>

// yaml
#include <yaml-cpp/yaml.h>

namespace snap {

//! Refuse a key of the map `node` that its reader does not read: a typo would
//! otherwise leave the setting at its default without a word. `path` names the
//! block in the message; a block shared with another library also lists the
//! keys that library reads (`other` names it).
void check_keys(YAML::Node const& node, std::string const& path,
                std::vector<std::string> const& keys,
                std::string const& other = "",
                std::vector<std::string> const& other_keys = {});

}  // namespace snap
