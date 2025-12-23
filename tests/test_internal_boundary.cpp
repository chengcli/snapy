// external
#include <gtest/gtest.h>
#include <yaml-cpp/yaml.h>

// torch
#include <torch/torch.h>

// snap
#include <snap/bc/internal_boundary.hpp>

#include "device_testing.hpp"

using namespace snap;

char const* bc_config = R"(
solid-density: 1.e3
solid-pressure: 1.9
max-iter: 5
)";

TEST_P(DeviceTest, mark_solid) {
  auto op = InternalBoundaryOptionsImpl::from_yaml(YAML::Load(bc_config));
  auto pib = InternalBoundary(op);

  auto w = torch::randn({5, 1, 5, 5}, torch::device(device).dtype(dtype));
  auto solid = torch::randn({1, 5, 5}, torch::device(device).dtype(dtype));
  auto wlr = torch::randn({2, 5, 1, 5, 5}, torch::device(device).dtype(dtype));

  solid.masked_fill_(solid > 0.5, 1);
  solid.masked_fill_(solid < 0.1, 0);

  solid = solid.to(torch::kBool);

  pib->mark_prim_solid_(w, solid);
  wlr = pib->forward(wlr, 3, solid);

  std::cout << w << std::endl;
  std::cout << solid << std::endl;
  std::cout << wlr << std::endl;
}

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);

  return RUN_ALL_TESTS();
}
