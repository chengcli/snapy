#include <gtest/gtest.h>
#include <yaml-cpp/yaml.h>

#include <snap/input/report.hpp>
#include <sstream>

namespace {
YAML::Node report(std::string const& section, YAML::Node const& input) {
  std::ostringstream text;
  snap::report_input(text, section, input);
  return YAML::Load(text.str())[section];
}
}  // namespace

TEST(InputReport, preserves_input_and_fills_each_output_independently) {
  auto input = YAML::Load("[{type: netcdf, combine: false}, {type: restart}]");
  auto result = report("outputs", input);
  ASSERT_EQ(result.size(), 2u);
  EXPECT_FALSE(result[0]["combine"].as<bool>());
  EXPECT_TRUE(result[1]["combine"].as<bool>());
  EXPECT_EQ(result[1]["type"].Scalar(), "restart");
  EXPECT_EQ(result[0]["dt"].as<double>(), 0.);
  EXPECT_FALSE(result[0]["fid"]);  // assigned in code
  EXPECT_FALSE(input[1]["combine"]);
}

TEST(InputReport, respects_nested_aliases_and_optional_sections) {
  auto geo = report("geometry", YAML::Load("{bounds: {x2max_pi: 1}}"));
  EXPECT_TRUE(geo["bounds"]["x2max_pi"]);
  EXPECT_FALSE(geo["bounds"]["x2max"]);
  EXPECT_TRUE(geo["cells"]["nghost"]);
  EXPECT_FALSE(geo["cells"]["x1rat"]);

  auto forcing = report("forcing", YAML::Load("{coriolis: {omega1: 2}}"));
  EXPECT_EQ(forcing["coriolis"]["omega1"].as<double>(), 2.);
  EXPECT_EQ(forcing["coriolis"]["omega2"].as<double>(), 0.);
  EXPECT_FALSE(forcing["diffusion"]);
  auto extra = report("integration", YAML::Load("{execution: eager}"));
  EXPECT_EQ(extra["execution"].Scalar(), "eager");
}

TEST(InputReport, omitted_core_section_uses_only_deck_defaults) {
  YAML::Node const deck = YAML::Load("{}");
  auto layout = report("distribute", deck["distribute"]);
  EXPECT_EQ(layout["blocks_per_process"].as<int>(), 1);
  EXPECT_EQ(layout["layout"].Scalar(), "slab");
  EXPECT_FALSE(layout["device"]);  // supplied by the runtime
  EXPECT_FALSE(deck["distribute"]);
}
