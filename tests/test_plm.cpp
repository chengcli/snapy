// C/C++
#include <algorithm>
#include <cmath>

// external
#include <gtest/gtest.h>

// torch
#include <torch/torch.h>

// fvm
#include <snap/recon/interp_simple.hpp>
#include <snap/recon/interpolation.hpp>

// tests
#include "device_testing.hpp"

using namespace snap;

TEST_P(DeviceTest, interp_plm) {
  double phim1 = 1.0;
  double phi = 2.0;
  double phip1 = 3.0;
  double result = interp_plm(phim1, phi, phip1);
  double expected_result = 1.5;
  EXPECT_NEAR(result, expected_result, 1.E-10);
}

TEST_P(DeviceTest, interp_plm_torch1) {
  PLMInterp interp;
  interp->to(device, dtype);

  for (int i = 0; i < 10; ++i) {
    torch::Tensor phi = torch::randn({3}, torch::device(device).dtype(dtype));
    if (dtype == torch::kFloat32) {
      auto result1l = interp_plm(phi[0].item<float>(), phi[1].item<float>(),
                                 phi[2].item<float>());
      auto result1r = interp_plm(phi[2].item<float>(), phi[1].item<float>(),
                                 phi[0].item<float>());

      auto [resultl, resultr] = interp->forward(phi, 0);

      EXPECT_NEAR(result1l, resultl.item<float>(), 2.E-6);
      EXPECT_NEAR(result1r, resultr.item<float>(), 2.E-6);
    } else {
      auto result1l = interp_plm(phi[0].item<double>(), phi[1].item<double>(),
                                 phi[2].item<double>());
      auto result1r = interp_plm(phi[2].item<double>(), phi[1].item<double>(),
                                 phi[0].item<double>());

      auto [resultl, resultr] = interp->forward(phi, 0);
      EXPECT_NEAR(result1l, resultl.item<double>(), 1.E-6);
      EXPECT_NEAR(result1r, resultr.item<double>(), 1.E-6);
    }
  }
}

TEST_P(DeviceTest, interp_plm_torch2) {
  PLMInterp interp;
  interp->to(device, dtype);

  for (int i = 0; i < 10; ++i) {
    torch::Tensor phi =
        torch::randn({2, 3}, torch::device(device).dtype(dtype));
    if (dtype == torch::kFloat32) {
      auto result1l =
          interp_plm(phi[0][0].item<float>(), phi[0][1].item<float>(),
                     phi[0][2].item<float>());
      auto result1r =
          interp_plm(phi[0][2].item<float>(), phi[0][1].item<float>(),
                     phi[0][0].item<float>());
      auto result2l =
          interp_plm(phi[1][0].item<float>(), phi[1][1].item<float>(),
                     phi[1][2].item<float>());
      auto result2r =
          interp_plm(phi[1][2].item<float>(), phi[1][1].item<float>(),
                     phi[1][0].item<float>());

      auto [resultl, resultr] = interp->forward(phi, 1);

      EXPECT_NEAR(result1l, resultl[0].item<float>(), 2.E-6);
      EXPECT_NEAR(result1r, resultr[0].item<float>(), 2.E-6);

      EXPECT_NEAR(result2l, resultl[1].item<float>(), 2.E-6);
      EXPECT_NEAR(result2r, resultr[1].item<float>(), 2.E-6);
    } else {
      auto result1l =
          interp_plm(phi[0][0].item<double>(), phi[0][1].item<double>(),
                     phi[0][2].item<double>());
      auto result1r =
          interp_plm(phi[0][2].item<double>(), phi[0][1].item<double>(),
                     phi[0][0].item<double>());
      auto result2l =
          interp_plm(phi[1][0].item<double>(), phi[1][1].item<double>(),
                     phi[1][2].item<double>());
      auto result2r =
          interp_plm(phi[1][2].item<double>(), phi[1][1].item<double>(),
                     phi[1][0].item<double>());

      auto [resultl, resultr] = interp->forward(phi, 1);

      EXPECT_NEAR(result1l, resultl[0].item<double>(), 2.E-6);
      EXPECT_NEAR(result1r, resultr[0].item<double>(), 2.E-6);

      EXPECT_NEAR(result2l, resultl[1].item<double>(), 2.E-6);
      EXPECT_NEAR(result2r, resultr[1].item<double>(), 2.E-6);
    }
  }
}

TEST_P(DeviceTest, interp_plm_torch3) {
  PLMInterp interp;
  interp->to(device, dtype);

  for (int i = 0; i < 10; ++i) {
    torch::Tensor phi =
        torch::randn({3, 2}, torch::device(device).dtype(dtype));
    if (dtype == torch::kFloat32) {
      auto result1l =
          interp_plm(phi[0][0].item<float>(), phi[1][0].item<float>(),
                     phi[2][0].item<float>());

      auto result1r =
          interp_plm(phi[2][0].item<float>(), phi[1][0].item<float>(),
                     phi[0][0].item<float>());

      auto result2l =
          interp_plm(phi[0][1].item<float>(), phi[1][1].item<float>(),
                     phi[2][1].item<float>());

      auto result2r =
          interp_plm(phi[2][1].item<float>(), phi[1][1].item<float>(),
                     phi[0][1].item<float>());

      auto [resultl, resultr] = interp->forward(phi, 0);

      EXPECT_NEAR(result1l, resultl[0][0].item<float>(), 2.E-6);
      EXPECT_NEAR(result1r, resultr[0][0].item<float>(), 2.E-6);

      EXPECT_NEAR(result2l, resultl[0][1].item<float>(), 2.E-6);
      EXPECT_NEAR(result2r, resultr[0][1].item<float>(), 2.E-6);
    } else {
      auto result1l =
          interp_plm(phi[0][0].item<double>(), phi[1][0].item<double>(),
                     phi[2][0].item<double>());

      auto result1r =
          interp_plm(phi[2][0].item<double>(), phi[1][0].item<double>(),
                     phi[0][0].item<double>());

      auto result2l =
          interp_plm(phi[0][1].item<double>(), phi[1][1].item<double>(),
                     phi[2][1].item<double>());

      auto result2r =
          interp_plm(phi[2][1].item<double>(), phi[1][1].item<double>(),
                     phi[0][1].item<double>());

      auto [resultl, resultr] = interp->forward(phi, 0);

      EXPECT_NEAR(result1l, resultl[0][0].item<double>(), 2.E-6);
      EXPECT_NEAR(result1r, resultr[0][0].item<double>(), 2.E-6);

      EXPECT_NEAR(result2l, resultl[0][1].item<double>(), 2.E-6);
      EXPECT_NEAR(result2r, resultr[0][1].item<double>(), 2.E-6);
    }
  }
}

int main(int argc, char **argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
