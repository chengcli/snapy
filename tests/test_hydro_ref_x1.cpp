// C/C++
#include <limits>
#include <vector>

// gtest
#include <gtest/gtest.h>

// torch
#include <torch/torch.h>

// snap
#include <snap/snap.h>

#include <snap/hydro/hydro_dispatch.hpp>

namespace {

struct RefOutput {
  torch::Tensor psf_lo;
  torch::Tensor psf_hi;
  torch::Tensor pref;
  torch::Tensor dsf;
  torch::Tensor dref;
};

RefOutput allocate_output(torch::Tensor const& w) {
  auto sizes = w.sizes().slice(1).vec();
  return {torch::empty(sizes, w.options()), torch::empty(sizes, w.options()),
          torch::empty(sizes, w.options()), torch::empty(sizes, w.options()),
          torch::empty(sizes, w.options())};
}

RefOutput tensor_reference(torch::Tensor const& w, torch::Tensor const& dx1f,
                           torch::Tensor const& anchor_in,
                           torch::Tensor const& gam, int is, int iu,
                           double grav, bool uniform, bool phys_in,
                           bool phys_out) {
  int nc1 = w.size(-1);
  auto rho = w[snap::IDN];
  auto dp = grav * rho * dx1f;
  auto cum = torch::cumsum(dp, -1);
  auto anchor =
      anchor_in.defined()
          ? anchor_in
          : (w[snap::IPR].select(-1, iu) *
             torch::exp(-grav * 0.5 * dx1f[iu] /
                        (w[snap::IPR].select(-1, iu) / rho.select(-1, iu))))
                .unsqueeze(-1);

  auto out = allocate_output(w);
  out.psf_lo.copy_(anchor + cum.select(-1, iu).unsqueeze(-1) - cum + dp);
  out.psf_hi.copy_(out.psf_lo - dp);
  out.psf_lo.clamp_min_(std::numeric_limits<double>::min());
  out.psf_hi.clamp_min_(std::numeric_limits<double>::min());

  if (uniform) {
    out.pref.copy_(0.5 * (out.psf_lo + out.psf_hi));
    auto faces =
        torch::cat({out.psf_lo, out.psf_hi.narrow(-1, nc1 - 1, 1)}, -1);
    constexpr double w6[6] = {11. / 1440., -31. / 480., 401. / 720.,
                              401. / 720., -31. / 480., 11. / 1440.};
    auto six = w6[0] * faces.narrow(-1, 0, nc1 - 4);
    for (int k = 1; k < 6; ++k) {
      six += w6[k] * faces.narrow(-1, k, nc1 - 4);
    }
    auto lo = torch::minimum(out.psf_lo, out.psf_hi).narrow(-1, 2, nc1 - 4);
    auto hi = torch::maximum(out.psf_lo, out.psf_hi).narrow(-1, 2, nc1 - 4);
    auto mid = out.pref.narrow(-1, 2, nc1 - 4);
    mid.copy_(torch::where((six >= lo) & (six <= hi), six, mid));

    constexpr double w6e[2][6] = {
        {95. / 288., 1427. / 1440., -133. / 240., 241. / 720., -173. / 1440.,
         3. / 160.},
        {-3. / 160., 637. / 1440., 511. / 720., -43. / 240., 77. / 1440.,
         -11. / 1440.},
    };
    if (!phys_in) {
      for (int j : {0, 1}) {
        auto val = w6e[j][0] * faces.select(-1, 0);
        for (int m = 1; m < 6; ++m) {
          val += w6e[j][m] * faces.select(-1, m);
        }
        auto flo =
            torch::minimum(out.psf_lo.select(-1, j), out.psf_hi.select(-1, j));
        auto fhi =
            torch::maximum(out.psf_lo.select(-1, j), out.psf_hi.select(-1, j));
        auto cur = out.pref.select(-1, j);
        cur.copy_(torch::where((val >= flo) & (val <= fhi), val, cur));
      }
    }
    if (!phys_out) {
      for (int j : {nc1 - 2, nc1 - 1}) {
        int sigma = j - (nc1 - 5);
        auto val = w6e[4 - sigma][5] * faces.select(-1, nc1 - 5);
        for (int m = 1; m < 6; ++m) {
          val += w6e[4 - sigma][5 - m] * faces.select(-1, nc1 - 5 + m);
        }
        auto flo =
            torch::minimum(out.psf_lo.select(-1, j), out.psf_hi.select(-1, j));
        auto fhi =
            torch::maximum(out.psf_lo.select(-1, j), out.psf_hi.select(-1, j));
        auto cur = out.pref.select(-1, j);
        cur.copy_(torch::where((val >= flo) & (val <= fhi), val, cur));
      }
    }
  } else {
    auto ratio = out.psf_lo / out.psf_hi;
    out.pref.copy_(torch::where((ratio - 1.).abs() < 1.e-6,
                                0.5 * (out.psf_lo + out.psf_hi),
                                dp / torch::log(ratio)));
  }

  auto kbot = w[snap::IPR].select(-1, is).unsqueeze(-1) /
              rho.select(-1, is).unsqueeze(-1).pow(gam);
  out.dref.copy_((out.pref / kbot).pow(1.0 / gam));
  out.dsf.copy_((out.psf_lo / kbot).pow(1.0 / gam));
  return out;
}

struct Inputs {
  torch::Tensor w;
  torch::Tensor dx1f;
  torch::Tensor anchor;
  torch::Tensor gam;
};

Inputs make_inputs(torch::ScalarType dtype, bool uniform, bool with_anchor) {
  auto options = torch::TensorOptions().dtype(dtype).device(torch::kCPU);
  constexpr int nc3 = 2;
  constexpr int nc2 = 3;
  constexpr int nc1 = 10;
  auto w = torch::zeros({snap::IPR + 1, nc3, nc2, nc1}, options);
  auto x = torch::arange(nc1, options).view({1, 1, nc1});
  auto column = torch::arange(nc3 * nc2, options).view({nc3, nc2, 1});
  w[snap::IDN].copy_(1.0 + 0.01 * x + 0.002 * column);
  w[snap::IPR].copy_(100.0 - 0.15 * x + 0.03 * column);
  auto dx1f = uniform ? torch::full({nc1}, 0.1, options)
                      : torch::linspace(0.08, 0.13, nc1, options);
  auto gam = torch::full({nc3, nc2, 1}, 1.4, options);
  torch::Tensor anchor;
  if (with_anchor) {
    anchor = (95.0 + 0.02 * column).contiguous();
  }
  return {w, dx1f, anchor, gam};
}

void dispatch(Inputs const& in, RefOutput const& out, bool uniform,
              bool phys_in, bool phys_out) {
  at::native::call_hydro_ref_x1(in.w.device().type(), in.w, in.dx1f, in.anchor,
                                in.gam, torch::Tensor(), out.psf_lo, out.psf_hi,
                                out.pref, out.dsf, out.dref, 2, 7, 1.0, uniform,
                                phys_in, phys_out);
}

void expect_close(RefOutput const& actual, RefOutput const& expected,
                  double rtol, double atol) {
  EXPECT_TRUE(torch::allclose(actual.psf_lo, expected.psf_lo, rtol, atol));
  EXPECT_TRUE(torch::allclose(actual.psf_hi, expected.psf_hi, rtol, atol));
  EXPECT_TRUE(torch::allclose(actual.pref, expected.pref, rtol, atol));
  EXPECT_TRUE(torch::allclose(actual.dsf, expected.dsf, rtol, atol));
  EXPECT_TRUE(torch::allclose(actual.dref, expected.dref, rtol, atol));
}

TEST(HydroRefX1Dispatch, cpu_matches_tensor_reference) {
  for (auto dtype : {torch::kFloat32, torch::kFloat64}) {
    for (bool uniform : {false, true}) {
      for (bool with_anchor : {false, true}) {
        auto in = make_inputs(dtype, uniform, with_anchor);
        bool physical = !with_anchor;
        auto expected = tensor_reference(in.w, in.dx1f, in.anchor, in.gam, 2, 7,
                                         1.0, uniform, physical, physical);
        auto actual = allocate_output(in.w);
        dispatch(in, actual, uniform, physical, physical);
        double tolerance = dtype == torch::kFloat32 ? 2.e-5 : 2.e-12;
        expect_close(actual, expected, tolerance, tolerance);
      }
    }
  }
}

TEST(HydroRefX1Dispatch, cuda_matches_cpu) {
  if (!torch::cuda::is_available()) {
    GTEST_SKIP() << "CUDA is not available";
  }

  for (bool uniform : {false, true}) {
    auto cpu_in = make_inputs(torch::kFloat64, uniform, true);
    auto expected = allocate_output(cpu_in.w);
    dispatch(cpu_in, expected, uniform, false, false);

    Inputs gpu_in = {cpu_in.w.to(torch::kCUDA), cpu_in.dx1f.to(torch::kCUDA),
                     cpu_in.anchor.to(torch::kCUDA),
                     cpu_in.gam.to(torch::kCUDA)};
    auto actual = allocate_output(gpu_in.w);
    dispatch(gpu_in, actual, uniform, false, false);
    expect_close({actual.psf_lo.cpu(), actual.psf_hi.cpu(), actual.pref.cpu(),
                  actual.dsf.cpu(), actual.dref.cpu()},
                 expected, 2.e-12, 2.e-12);
  }
}

}  // namespace
