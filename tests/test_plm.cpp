// C/C++
#include <algorithm>
#include <cmath>

// external
#include <gtest/gtest.h>

// torch
#include <torch/torch.h>

// base
#include <globals.h>

// fvm
#include <fvm/recon/interp_simple.hpp>
#include <fvm/recon/interpolation.hpp>

// tests
#include "device_testing.hpp"

using namespace canoe;

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

      auto result2 = interp->forward(phi);

      EXPECT_NEAR(result1l, result2[0].item<float>(), 2.E-6);
      EXPECT_NEAR(result1r, result2[1].item<float>(), 2.E-6);
    } else {
      auto result1l = interp_plm(phi[0].item<double>(), phi[1].item<double>(),
                                 phi[2].item<double>());
      auto result1r = interp_plm(phi[2].item<double>(), phi[1].item<double>(),
                                 phi[0].item<double>());

      auto result2 = interp->forward(phi);
      EXPECT_NEAR(result1l, result2[0].item<double>(), 1.E-6);
      EXPECT_NEAR(result1r, result2[1].item<double>(), 1.E-6);
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

      auto result = interp->forward(phi);

      EXPECT_NEAR(result1l, result[0][0].item<float>(), 2.E-6);
      EXPECT_NEAR(result1r, result[1][0].item<float>(), 2.E-6);

      EXPECT_NEAR(result2l, result[0][1].item<float>(), 2.E-6);
      EXPECT_NEAR(result2r, result[1][1].item<float>(), 2.E-6);
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

      auto result = interp->forward(phi);

      EXPECT_NEAR(result1l, result[0][0].item<double>(), 2.E-6);
      EXPECT_NEAR(result1r, result[1][0].item<double>(), 2.E-6);

      EXPECT_NEAR(result2l, result[0][1].item<double>(), 2.E-6);
      EXPECT_NEAR(result2r, result[1][1].item<double>(), 2.E-6);
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

      auto phiu = phi.unfold(0, 3, 1);
      auto result = interp->forward(phiu);

      EXPECT_NEAR(result1l, result[0][0][0].item<float>(), 2.E-6);
      EXPECT_NEAR(result1r, result[1][0][0].item<float>(), 2.E-6);

      EXPECT_NEAR(result2l, result[0][0][1].item<float>(), 2.E-6);
      EXPECT_NEAR(result2r, result[1][0][1].item<float>(), 2.E-6);
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

      auto phiu = phi.unfold(0, 3, 1);
      auto result = interp->forward(phiu);

      EXPECT_NEAR(result1l, result[0][0][0].item<double>(), 2.E-6);
      EXPECT_NEAR(result1r, result[1][0][0].item<double>(), 2.E-6);

      EXPECT_NEAR(result2l, result[0][0][1].item<double>(), 2.E-6);
      EXPECT_NEAR(result2r, result[1][0][1].item<double>(), 2.E-6);
    }
  }
}

int main(int argc, char **argv) {
  testing::InitGoogleTest(&argc, argv);
  start_logging(argc, argv);

  return RUN_ALL_TESTS();
}
