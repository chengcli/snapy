// C/C++
#include <fstream>
#include <string>

// gtest
#include <gtest/gtest.h>

// yaml
#include <yaml-cpp/yaml.h>

// snap
#include <snap/mesh/meshblock.hpp>

using namespace snap;

namespace {

// kintera's species table is process-global, so one card for the whole binary
constexpr char const* kCard = "test_cycle_diagnostics.yaml";

// Load `card` as a run does; the error message, or "" when it loads.
std::string load_error(YAML::Node const& card, std::string const& name) {
  {
    std::ofstream out(name);
    out << card;
  }
  try {
    MeshBlockOptionsImpl::from_yaml(name);
  } catch (std::exception const& e) {
    return e.what();
  }
  return "";
}

// A misspelled key must be refused, and the message must name it by path.
void expect_refused(std::string const& block, std::string const& key) {
  auto card = YAML::LoadFile(kCard);
  auto node = card;
  std::string rest = block;
  for (auto pos = rest.find('/'); !rest.empty(); pos = rest.find('/')) {
    auto part = rest.substr(0, pos);
    node.reset(node[part]);
    rest = pos == std::string::npos ? "" : rest.substr(pos + 1);
  }
  node[key] = 1;
  auto msg = load_error(card, "test_yaml_keys_card.yaml");
  EXPECT_NE(msg.find("unknown key '" + block + "/" + key + "'"),
            std::string::npos)
      << "loaded or failed otherwise: '" << msg << "'";
}

}  // namespace

TEST(yaml_keys, the_card_itself_loads) {
  EXPECT_EQ(load_error(YAML::LoadFile(kCard), "test_yaml_keys_card.yaml"), "");
}

TEST(yaml_keys, const_gravity_refuses_an_unknown_key) {
  expect_refused("forcing/const-gravity", "garv1");
}

TEST(yaml_keys, geometry_cells_refuses_an_unknown_key) {
  expect_refused("geometry/cells", "nx4");
}

TEST(yaml_keys, external_boundaries_refuse_an_unknown_key) {
  expect_refused("boundary-condition/external", "x1-iner");
}
