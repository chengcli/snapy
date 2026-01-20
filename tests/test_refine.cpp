// external
#include <gtest/gtest.h>
#include <yaml-cpp/yaml.h>

// torch
#include <torch/torch.h>

// snapy
#include <snap/snap.h>

#include <snap/utils/refine.hpp>

// tests
#include "device_testing.hpp"

using namespace snap;

TEST_P(DeviceTest, refine_funcs) {
  int nc1 = 2;
  int nc2 = 3;
  int nc3 = 3;
  int nvar = 1;

  auto x = torch::empty({nvar, nc3, nc2, nc1}, torch::dtype(dtype));

  for (int n = 0; n < nvar; ++n)
    for (int k = 0; k < nc3; ++k)
      for (int j = 0; j < nc2; ++j)
        for (int i = 0; i < nc1; ++i) {
          x[n][k][j][i] = static_cast<float>(n + k + j + i + 1);
        }

  x = x.to(device);
  auto y = conservative_refine(x);
  auto z = conservative_coarsen(y);

  EXPECT_TRUE(torch::allclose(x, z, 1.E-6, 1.E-6));
}

int main(int argc, char **argv) {
  testing::InitGoogleTest(&argc, argv);

  return RUN_ALL_TESTS();
}
