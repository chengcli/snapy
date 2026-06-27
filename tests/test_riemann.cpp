// C/C++
#include <fstream>
#include <iostream>
#include <regex>
#include <sstream>

// external
#include <gtest/gtest.h>
#include <yaml-cpp/yaml.h>

// snap
#include <snap/eos/equation_of_state.hpp>
#include <snap/mesh/meshblock.hpp>
#include <snap/recon/reconstruct.hpp>
#include <snap/riemann/riemann_solver.hpp>

// tests
#include "device_testing.hpp"

enum {
  DIM1 = 3,
  DIM2 = 2,
  DIM3 = 1,
};

const char *block_config = R"(
reference-state:
  Tref: 300.
  Pref: 1.e5

species:
  - name: dry
    composition: {O: 0.42, N: 1.56, Ar: 0.01}
    cv_R: 2.5

dynamics:
  equation-of-state:
    type: moist-mixture
    density-floor:  1.e-10
    pressure-floor: 1.e-10
    limiter: false

  reconstruct:
    vertical: {type: weno5, scale: false, shock: false}
    horizontal: {type: weno5, scale: false, shock: false}

  riemann-solver:
    type: lmars

geometry:
  type: cartesian
  bounds: {x1min: 0., x1max: 1., x2min: 0., x2max: 1., x3min: 0., x3max: 1.}
  cells: {nx1: 200, nx2: 200, nx3: 200, nghost: 3}

boundary-condition:
  external:
    x1-inner: reflecting
    x1-outer: reflecting
    x2-inner: reflecting
    x2-outer: reflecting
    x3-inner: reflecting
    x3-outer: reflecting
)";

const char *small_ideal_gas_config = R"(
reference-state:
  Tref: 300.
  Pref: 1.e5

species:
  - name: dry
    composition: {O: 0.42, N: 1.56, Ar: 0.01}
    cv_R: 2.5

dynamics:
  equation-of-state:
    type: ideal-gas
    gammad: 1.4
    density-floor:  1.e-10
    pressure-floor: 1.e-10
    limiter: false

  reconstruct:
    vertical: {type: weno5, scale: false, shock: false}
    horizontal: {type: weno5, scale: false, shock: false}

  riemann-solver:
    type: lmars

geometry:
  type: cartesian
  bounds: {x1min: 0., x1max: 1., x2min: 0., x2max: 1., x3min: 0., x3max: 1.}
  cells: {nx1: 10, nx2: 8, nx3: 6, nghost: 3}

boundary-condition:
  external:
    x1-inner: reflecting
    x1-outer: reflecting
    x2-inner: reflecting
    x2-outer: reflecting
    x3-inner: reflecting
    x3-outer: reflecting
)";

using namespace snap;

TEST(HydroOptions, parses_fused_recon_riemann_flag) {
  auto config = std::regex_replace(
      block_config, std::regex("riemann-solver:\\n    type: lmars"),
      "fused-recon-riemann: true\n\n  riemann-solver:\n    type: lmars");

  char fname[80] = "/tmp/tempfile.XXXXXX";
  mkstemp(fname);
  std::ofstream outfile(fname);
  outfile << config;
  outfile.close();

  auto op_block = MeshBlockOptionsImpl::from_yaml(fname);
  EXPECT_TRUE(op_block->hydro()->fused_recon_riemann());

  std::remove(fname);
}

TEST(HydroOptions, ignores_old_fused_reconstruction_riemann_flag) {
  auto config = std::regex_replace(
      block_config, std::regex("riemann-solver:\\n    type: lmars"),
      "fused-reconstruction-riemann: true\n\n  riemann-solver:\n    type: "
      "lmars");

  char fname[80] = "/tmp/tempfile.XXXXXX";
  mkstemp(fname);
  std::ofstream outfile(fname);
  outfile << config;
  outfile.close();

  auto op_block = MeshBlockOptionsImpl::from_yaml(fname);
  EXPECT_FALSE(op_block->hydro()->fused_recon_riemann());

  std::remove(fname);
}

TEST_P(DeviceTest, fused_recon_riemann_matches_staged_ideal_gas_lmars) {
  if (device.type() != torch::kCUDA) {
    GTEST_SKIP() << "fused reconstruction/Riemann path is CUDA-only";
  }

  char staged_name[80] = "/tmp/tempfile.XXXXXX";
  mkstemp(staged_name);
  std::ofstream staged_file(staged_name);
  staged_file << small_ideal_gas_config;
  staged_file.close();

  auto fused_config = std::regex_replace(
      std::string(small_ideal_gas_config),
      std::regex("riemann-solver:\\n    type: lmars"),
      "fused-recon-riemann: true\n\n  riemann-solver:\n    type: lmars");
  char fused_name[80] = "/tmp/tempfile.XXXXXX";
  mkstemp(fused_name);
  std::ofstream fused_file(fused_name);
  fused_file << fused_config;
  fused_file.close();

  auto staged_opts = MeshBlockOptionsImpl::from_yaml(staged_name);
  auto fused_opts = MeshBlockOptionsImpl::from_yaml(fused_name);
  auto staged_block = MeshBlock(staged_opts);
  auto fused_block = MeshBlock(fused_opts);
  staged_block->to(device, dtype);
  fused_block->to(device, dtype);

  int nc1 = staged_block->pcoord->options->nc1();
  int nc2 = staged_block->pcoord->options->nc2();
  int nc3 = staged_block->pcoord->options->nc3();
  int nvar = staged_block->phydro->peos->nvar();

  torch::manual_seed(202);
  auto w =
      torch::rand({nvar, nc3, nc2, nc1}, torch::device(device).dtype(dtype));
  w[IDN].add_(1.0);
  w[IPR].add_(1.0);
  w.narrow(0, IVX, 3).sub_(0.5).mul_(0.1);

  auto u = staged_block->phydro->peos->compute("W->U", {w});
  Variables staged_vars, fused_vars;
  staged_vars["hydro_w"] = torch::empty_like(w);
  fused_vars["hydro_w"] = torch::empty_like(w);

  auto staged_du = staged_block->phydro->forward(1.e-4, u.clone(), staged_vars);
  auto fused_du = fused_block->phydro->forward(1.e-4, u.clone(), fused_vars);
  EXPECT_TRUE(torch::allclose(fused_du, staged_du, 1.e-8, 1.e-8));

  std::remove(staged_name);
  std::remove(fused_name);
}

TEST_P(DeviceTest, test_lmars) {
  // create a temporary file
  char fname[80] = "/tmp/tempfile.XXXXXX";
  mkstemp(fname);
  std::ofstream outfile(fname);
  outfile << block_config;
  outfile.close();

  auto op_block = MeshBlockOptionsImpl::from_yaml(fname);
  auto block = MeshBlock(op_block);
  block->to(device, dtype);

  auto pcoord = block->pcoord;
  auto phydro = block->phydro;
  auto peos = phydro->peos;
  auto precon = phydro->precon1;
  auto prsolver = phydro->priemann;

  int nc1 = pcoord->options->nc1();
  int nc2 = pcoord->options->nc2();
  int nc3 = pcoord->options->nc3();
  int nvar = peos->nvar();

  auto w =
      torch::randn({nvar, nc3, nc2, nc1}, torch::device(device).dtype(dtype))
          .abs();

  std::cout << "w.sizes(): " << w.sizes() << std::endl;

  auto start = std::chrono::high_resolution_clock::now();

  auto wlr = precon->forward(w, DIM1);

  auto flux = torch::zeros_like(wlr[0]);
  prsolver->forward(wlr[0], wlr[1], DIM1, flux);
  std::cout << "flux.sizes(): " << flux.sizes() << std::endl;

  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> elapsed = end - start;
  std::cout << "Time taken by test body: " << elapsed.count() << " seconds"
            << std::endl;
  std::remove(fname);
}

TEST_P(DeviceTest, test_hllc) {
  auto config = std::regex_replace(block_config, std::regex("lmars"), "hllc");
  char fname[80] = "/tmp/tempfile.XXXXXX";
  mkstemp(fname);
  std::ofstream outfile(fname);
  outfile << config;
  outfile.close();

  auto op_block = MeshBlockOptionsImpl::from_yaml(fname);
  auto block = MeshBlock(op_block);
  block->to(device, dtype);

  auto pcoord = block->pcoord;
  auto phydro = block->phydro;
  auto peos = phydro->peos;
  auto precon = phydro->precon1;
  auto prsolver = phydro->priemann;

  int nc1 = pcoord->options->nc1();
  int nc2 = pcoord->options->nc2();
  int nc3 = pcoord->options->nc3();
  int nvar = peos->nvar();

  auto w =
      torch::randn({nvar, nc3, nc2, nc1}, torch::device(device).dtype(dtype))
          .abs();

  std::cout << "w.sizes(): " << w.sizes() << std::endl;

  auto start = std::chrono::high_resolution_clock::now();

  auto wlr = precon->forward(w, DIM1);

  auto flux = torch::zeros_like(wlr[0]);
  prsolver->forward(wlr[0], wlr[1], DIM1, flux);
  std::cout << "flux.sizes(): " << flux.sizes() << std::endl;

  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> elapsed = end - start;
  std::cout << "Time taken by test body: " << elapsed.count() << " seconds"
            << std::endl;
  std::remove(fname);
}

int main(int argc, char **argv) {
  testing::InitGoogleTest(&argc, argv);

  return RUN_ALL_TESTS();
}
