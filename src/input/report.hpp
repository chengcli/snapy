#pragma once

#include <iosfwd>
#include <string>

namespace YAML {
class Node;
}

namespace snap {

//! Write an input section, adding literal schema defaults for omitted keys.
void report_input(std::ostream& os, std::string const& section,
                  YAML::Node const& input);

}  // namespace snap
