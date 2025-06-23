// external
#include <gtest/gtest.h>

// torch
#include <torch/torch.h>

// base
#include <globals.h>

// fvm
#include <fvm/bc/internal_boundary.hpp>

#include "device_testing.hpp"

TEST_P(DeviceTest, mark_solid) {
  auto pib = canoe::InternalBoundary(canoe::InternalBoundaryOptions());

  auto w = torch::randn({5, 1, 5, 5}, torch::device(device).dtype(dtype));
  auto solid = torch::randn({1, 5, 5}, torch::device(device).dtype(dtype));
  auto wlr = torch::randn({2, 5, 1, 5, 5}, torch::device(device).dtype(dtype));

  solid.masked_fill_(solid > 0.5, 1);
  solid.masked_fill_(solid < 0.1, 0);

  solid = solid.abs();

  auto w1 = pib->mark_solid(w, solid);
  wlr = pib->forward(wlr, 3, solid);

  // std::cout << w << std::endl;
  std::cout << solid << std::endl;
  std::cout << w1 << std::endl;
  std::cout << wlr << std::endl;
}

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  start_logging(argc, argv);

  return RUN_ALL_TESTS();
}
