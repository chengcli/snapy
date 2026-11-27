#include "report.hpp"

#include <yaml-cpp/yaml.h>

#include <ostream>

namespace snap {

// Built from src/options/*.yaml and linked into the library.
char const* input_report_schema(std::string const& section);

namespace {

// Fill a missing leaf without replacing a supplied value or its alias.
void add_default(YAML::Node map, std::string const& path,
                 YAML::Node const& row) {
  if (!map.IsMap()) return;
  auto slash = path.find('/');
  if (slash != std::string::npos) {
    auto key = path.substr(0, slash);
    if (!map[key]) map[key] = YAML::Node(YAML::NodeType::Map);
    add_default(map[key], path.substr(slash + 1), row);
    return;
  }
  if (map[path]) return;
  for (auto alias : row["alias"])
    if (map[alias.first.as<std::string>()]) return;
  map[path] = YAML::Clone(row["default"]);
}

void fill_section(YAML::Node node, YAML::Node const& schema) {
  if (node.IsSequence() && schema["items"]) {
    for (auto item : node) fill_section(item, schema["items"]);
  }
  if (!node.IsMap()) return;
  for (auto row : schema["defaults"])
    add_default(node, row["key"].as<std::string>(), row);
  for (auto child : schema["children"]) {
    auto key = child.first.as<std::string>();
    if (node[key]) fill_section(node[key], child.second);
  }
}

}  // namespace

void report_input(std::ostream& os, std::string const& section,
                  YAML::Node const& input) {
  auto result =
      input.IsDefined() ? YAML::Clone(input) : YAML::Node(YAML::NodeType::Map);
  if (auto schema = input_report_schema(section))
    fill_section(result, YAML::Load(schema));
  YAML::Node report;
  report[section] = result;
  os << report << '\n';
}

}  // namespace snap
