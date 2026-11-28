#include <gtest/gtest.h>
#include <snap/snap.h>
#include <torch/torch.h>

#include <snap/recon/x1_reference.hpp>

#include "device_testing.hpp"
using snap::IDN;
using snap::IPR;

TEST_P(DeviceTest, split_reference_matches_total_nonlinear_operator) {
  auto opts = torch::TensorOptions().device(device).dtype(torch::kFloat64);
  for (int grid : {0, 1, 2}) {
    bool stretched = grid == 1;
    auto dx = grid != 0 ? torch::linspace(80., 120., 20, opts)
                        : torch::full({20}, 100., opts);
    snap::HydroReferenceX1Operator op(dx, 16, stretched, true, true, true);
    auto w = torch::zeros({5, 2, 1, 20}, opts);
    w[IPR] = torch::linspace(1.e5, 8.e4, 20, opts);
    w[IDN] = w[IPR] / 8.e4;
    w[IPR][1] *= 1.01;
    for (double amplitude : {0., 1.e-8, .2, -.5}) {
      auto dw = torch::zeros_like(w);
      dw[IPR] = amplitude * w[IPR] * torch::sin(torch::arange(20, opts));
      dw[IDN] = amplitude * w[IDN] * torch::cos(torch::arange(20, opts));
      dw = dw.to(dtype);
      snap::HydroReferenceX1 actual(torch::stack({w, dw}, -1)), expected(w);
      op.evaluate_split(w, dw, 10., {}, actual);
      if (grid == 0 && dw.scalar_type() == torch::kFloat32) {
        // Exercise a broadcast background column and a full three-D field.
        auto direct =
            op.uniform_split(w[IDN][0], w[IPR], dw[IDN], dw[IPR], 10.);
        ASSERT_TRUE(direct.defined());
        int row = 0;
        for (auto t : {actual.psf_lo, actual.psf_hi, actual.pref, actual.dsf,
                       actual.dref}) {
          EXPECT_TRUE(torch::allclose(direct[row++],
                                      t.select(-1, 0) + t.select(-1, 1), 2.e-14,
                                      1.e-30));
        }
        for (auto t : {actual.psf_lo, actual.pref})
          for (int lane : {0, 1})
            EXPECT_TRUE(torch::allclose(direct[row++], t.select(-1, lane),
                                        2.e-14, 1.e-30));
      }
      op.evaluate(w + dw, 10., {}, expected);
      for (auto pair : {std::make_pair(actual.psf_lo, expected.psf_lo),
                        std::make_pair(actual.psf_hi, expected.psf_hi),
                        std::make_pair(actual.pref, expected.pref),
                        std::make_pair(actual.dsf, expected.dsf),
                        std::make_pair(actual.dref, expected.dref)}) {
        EXPECT_TRUE(
            torch::allclose(pair.first.select(-1, 0) + pair.first.select(-1, 1),
                            pair.second, 3.e-13, 1.e-13));
        if (amplitude == 0.)
          EXPECT_EQ(pair.first.select(-1, 1).abs().max().item<double>(), 0.);
      }
    }
  }
}

TEST_P(DeviceTest, uniform_column_stencil_matches_packed_at_walls) {
  if (dtype != torch::kFloat32) GTEST_SKIP() << "FP32 increment specialization";
  auto opts = torch::TensorOptions().device(device).dtype(torch::kFloat64);
  for (int n : {8, 33, 128, 129}) {
    auto dx = torch::ones({n}, opts);
    auto w = torch::ones({IPR + 1, 2, 3, n}, opts);
    auto x = torch::arange(w[IDN].numel(), opts).reshape_as(w[IDN]);
    w[IDN] += 0.2 * torch::sin(x);
    w[IPR] = 1000. + x;
    auto dw = (w * 1.e-5).to(dtype);
    for (int walls = 0; walls < 4; ++walls) {
      snap::HydroReferenceX1Operator op(dx, n - 4, false, walls & 1, walls & 2,
                                        true);
      auto direct = op.uniform_split(w[IDN], w[IPR], dw[IDN], dw[IPR], 1.);
      if (n > 128) {
        EXPECT_FALSE(direct.defined());
        continue;
      }
      ASSERT_TRUE(direct.defined());
      snap::HydroReferenceX1 pairs;
      op.evaluate_split(w, dw, 1., {}, pairs);
      EXPECT_TRUE(torch::allclose(
          direct[3], pairs.dsf.select(-1, 0) + pairs.dsf.select(-1, 1), 1.e-13,
          1.e-15));
      EXPECT_TRUE(torch::allclose(
          direct[4], pairs.dref.select(-1, 0) + pairs.dref.select(-1, 1),
          1.e-13, 1.e-15));
    }
  }
}

TEST_P(DeviceTest, split_reference_retains_increments_below_fp64_ulp) {
  auto opts = torch::TensorOptions().device(device).dtype(torch::kFloat64);
  auto dx = torch::ones({14}, opts);
  snap::HydroReferenceX1Operator op(dx, 10, false, true, true, true);
  auto w = torch::zeros({5, 1, 1, 14}, opts);
  w[IDN].fill_(1.);
  w[IPR].fill_(1.e12);
  auto dw = torch::zeros_like(w);
  dw[IPR].fill_(1.e-8);
  dw = dw.to(dtype);
  EXPECT_TRUE(torch::equal(w + dw, w));
  snap::HydroReferenceX1 out(torch::stack({w, dw}, -1));
  op.evaluate_split(w, dw, 0., {}, out);
  EXPECT_TRUE(torch::allclose(out.psf_lo.select(-1, 1),
                              dw[IPR].to(torch::kFloat64), 1.e-14, 1.e-24));
  EXPECT_TRUE(torch::allclose(out.pref.select(-1, 1),
                              dw[IPR].to(torch::kFloat64), 1.e-14, 1.e-24));
  if (dtype == torch::kFloat32) {
    auto result = op.uniform_split(w[IDN], w[IPR], dw[IDN], dw[IPR], 0.);
    EXPECT_TRUE(torch::equal(result[6], out.psf_lo.select(-1, 1)));
    EXPECT_TRUE(
        torch::allclose(result[8], out.pref.select(-1, 1), 1.e-14, 1.e-24));
  }
  EXPECT_LT(out.dref.select(-1, 1).abs().max().item<double>(), 1.e-34);
}

TEST_P(DeviceTest,
       split_reference_floor_uses_current_total_and_drops_increment) {
  auto opts = torch::TensorOptions().device(device).dtype(torch::kFloat64);
  auto dx = torch::ones({14}, opts);
  snap::HydroReferenceX1Operator op(dx, 10, false, true, true, true);
  auto w = torch::zeros({5, 1, 1, 14}, opts);
  w[IDN].fill_(1.);
  w[IPR].fill_(2.);
  auto dw = torch::zeros_like(w);
  dw[IDN].fill_(.1);
  dw = dw.to(dtype);
  snap::HydroReferenceX1 out(torch::stack({w, dw}, -1)), expected(w);
  op.evaluate_split(w, dw, 10., {}, out);
  op.evaluate(w + dw, 10., {}, expected);
  EXPECT_TRUE(
      torch::allclose(out.psf_lo.select(-1, 0) + out.psf_lo.select(-1, 1),
                      expected.psf_lo, 1.e-14, 1.e-14));
  if (dtype == torch::kFloat32) {
    auto result = op.uniform_split(w[IDN], w[IPR], dw[IDN], dw[IPR], 10.);
    EXPECT_TRUE(torch::equal(
        result[1], out.psf_hi.select(-1, 0) + out.psf_hi.select(-1, 1)));
  }
  EXPECT_EQ(out.psf_hi.select(-1, 1).select(-1, 13).item<double>(), 0.);
}
