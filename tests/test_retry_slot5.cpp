// V3 X5 m1: slot 5 of the redo read comes from the register monitor's
// implicit.column flag (I-5) instead of the VIC latch solve_failed_tensor().
// The switch rests on the two being equal; this tests it on stepped decks:
// the partial (1) and full (9) vertical implicit solve, a clean column and
// one with a NaN planted in an interior cell before a step, on CPU and CUDA.
// After every stage the monitor flag and the latch must agree bitwise, and
// check_redo must reject exactly the steps the latch marks.

// external
#include <gtest/gtest.h>
#include <yaml-cpp/yaml.h>

// C/C++
#include <cstdio>
#include <fstream>
#include <limits>
#include <string>

// torch
#include <torch/torch.h>

// snap
#include <snap/snap.h>

#include <snap/mesh/meshblock.hpp>

// tests
#include "cuda_test_gate.hpp"

using namespace snap;

namespace {

MeshBlock column(int scheme, torch::Device device) {
  auto node = YAML::Load(R"(
geometry:
  type: cartesian
  cells: {nx1: 16, nx2: 4, nx3: 1, nghost: 3}
  bounds: {x1min: 0, x1max: 16, x2min: 0, x2max: 4, x3min: 0, x3max: 1}
dynamics:
  equation-of-state: {type: ideal-gas, gammad: 1.4, weight: 0.029}
  reconstruct:
    vertical: {type: plm}
    horizontal: {type: plm}
  riemann-solver: {type: lmars}
forcing:
  const-gravity: {grav1: -1, gravity-work: face, gravity-work-fixer: false}
integration: {type: rk3, implicit-scheme: 9, cfl: 0.5}
boundary-condition:
  external: {x1-inner: reflecting, x1-outer: reflecting,
             x2-inner: periodic, x2-outer: periodic,
             x3-inner: periodic, x3-outer: periodic}
)");
  node["integration"]["implicit-scheme"] = scheme;
  auto path = "test-retry-slot5-" + std::to_string(scheme) + ".yaml";
  {
    std::ofstream card(path);
    card << node;
  }
  auto block = MeshBlock(MeshBlockOptionsImpl::from_yaml(path));
  std::remove(path.c_str());
  block->to(device);
  return block;
}

//! the monitor's implicit.column flag and the VIC latch, as slot 5 reads them
std::pair<torch::Tensor, torch::Tensor> slot5(MeshBlock const& block,
                                              torch::TensorOptions opts) {
  auto const& mon = block->phydro->flux_register().monitor();
  auto flag =
      (mon.retry_flags[mon.retry_index("implicit.column")] > 0).to(opts).cpu();
  auto latch = block->phydro->picorr->solve_failed_tensor(opts).cpu();
  return {flag, latch};
}

void run(int scheme, torch::Device device, int nan_step) {
  auto block = column(scheme, device);
  auto c = block->pcoord->options;
  auto w = torch::zeros(
      {5, c->nc3(), c->nc2(), c->nc1()},
      torch::TensorOptions().dtype(torch::kFloat64).device(device));
  w[IDN].fill_(1.);
  w[IPR].fill_(1.e5);
  w[IVX] = 0.1 * torch::sin(block->pcoord->x1v).view({1, 1, -1});
  Variables vars{{"hydro_w", w}};
  block->initialize(vars);
  auto opts = vars.at("hydro_u").options();
  int nstage = static_cast<int>(block->pintg->stages.size());
  int rejected = 0;
  for (int n = 0; n < 6; ++n) {
    double dt = block->max_time_step(vars);
    if (n ==
        nan_step)  // an interior cell (3 ghosts), as a bad step would leave it
      vars.at("hydro_u")[IPR][0][4][10] =
          std::numeric_limits<double>::quiet_NaN();
    bool latched = false;
    for (int s = 0; s < nstage; ++s) {
      block->forward(vars, dt, s);
      auto [flag, latch] = slot5(block, opts);
      EXPECT_TRUE(torch::equal(flag, latch))
          << "scheme " << scheme << " step " << n << " stage " << s
          << ": monitor " << flag.item<double>() << " latch "
          << latch.item<double>();
      latched = latch.item<double>() > 0.;
    }
    int status = block->check_redo(vars);
    EXPECT_EQ(status, latched ? 1 : 0) << "scheme " << scheme << " step " << n;
    if (latched) {
      ++rejected;
      break;  // the restored step holds the planted NaN again
    }
  }
  if (nan_step >= 0) EXPECT_GE(rejected, 1) << "the planted NaN was not caught";
}

}  // namespace

TEST(RetrySlot5, partial_clean) { run(1, torch::kCPU, -1); }
TEST(RetrySlot5, full_clean) { run(9, torch::kCPU, -1); }
TEST(RetrySlot5, partial_nan) { run(1, torch::kCPU, 2); }
TEST(RetrySlot5, full_nan) { run(9, torch::kCPU, 2); }
TEST(RetrySlot5, cuda_partial_nan) {
  if (!snapy_cuda_test_enabled()) GTEST_SKIP() << "CPU build or no CUDA device";
  run(1, torch::kCUDA, 2);
}
TEST(RetrySlot5, cuda_full_nan) {
  if (!snapy_cuda_test_enabled()) GTEST_SKIP() << "CPU build or no CUDA device";
  run(9, torch::kCUDA, 2);
}
TEST(RetrySlot5, cuda_full_clean) {
  if (!snapy_cuda_test_enabled()) GTEST_SKIP() << "CPU build or no CUDA device";
  run(9, torch::kCUDA, -1);
}

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
