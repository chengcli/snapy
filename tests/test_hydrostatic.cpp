// C/C++
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>

// POSIX
#include <unistd.h>

// gtest
#include <gtest/gtest.h>

// snap
#include <snap/snap.h>

#include <snap/mesh/mesh.hpp>

using namespace snap;

namespace {

const char* cubed_sphere_hydrostatic_config = R"(
reference-state:
  Tref: 300.
  Pref: 100.

species:
  - name: dry
    composition: {O: 0.42, N: 1.56, Ar: 0.01}
    cv_R: 2.5

dynamics:
  equation-of-state:
    type: ideal-gas
    gammad: 1.4
    density-floor: 1.e-12
    pressure-floor: 1.e-12
    limiter: false

  reconstruct:
    vertical: {type: weno5, scale: false, shock: false}
    horizontal: {type: weno5, scale: false, shock: false}

  riemann-solver:
    type: lmars

integration:
  type: rk3
  cfl: 0.4
  implicit-scheme: 0

forcing:
  const-gravity:
    grav1: -1.
    non-hydrostatic: 0.

distribute:
  layout: cubed-sphere
  nb2: 1
  nb3: 1
  blocks_per_process: 6
  verbose: false

geometry:
  type: gnomonic-equiangle
  cells: {nx1: 24, nx2: 8, nx3: 8, nghost: 3}
  bounds:
    x1min: 10.
    x1max: 11.
    x2min_pi: -0.25
    x2max_pi: 0.25
    x3min_pi: -0.25
    x3max_pi: 0.25

boundary-condition:
  external:
    x1-inner: reflecting
    x1-outer: reflecting
    x2-inner: custom
    x2-outer: custom
    x3-inner: custom
    x3-outer: custom
)";

std::string write_temp_config() {
  char fname[] = "/tmp/test-hydrostatic-XXXXXX";
  int fd = mkstemp(fname);
  EXPECT_NE(fd, -1);
  if (fd != -1) close(fd);

  std::ofstream outfile(fname);
  outfile << cubed_sphere_hydrostatic_config;
  outfile.close();
  return fname;
}

void initialize_hydrostatic_column(MeshBlock const& block, Variables& vars) {
  constexpr double gamma = 1.4;
  constexpr double pressure0 = 100.;
  constexpr double density0 = 1.;
  constexpr double gravity = 1.;
  constexpr double radius0 = 10.;

  auto pcoord = block->pcoord;
  auto entropy = pressure0 / std::pow(density0, gamma);
  auto exponent = (gamma - 1.) / gamma;
  auto pressure_power = std::pow(pressure0, exponent) -
                        exponent * gravity * (pcoord->x1v - radius0) /
                            std::pow(entropy, 1. / gamma);
  auto pressure = pressure_power.pow(1. / exponent);
  auto density = (pressure / entropy).pow(1. / gamma);

  auto nc1 = pcoord->options->nc1();
  auto nc2 = pcoord->options->nc2();
  auto nc3 = pcoord->options->nc3();
  auto w = torch::zeros({block->phydro->peos->nvar(), nc3, nc2, nc1},
                        torch::kFloat64);
  w[IDN].copy_(density.view({1, 1, nc1}).expand({nc3, nc2, nc1}));
  w[IPR].copy_(pressure.view({1, 1, nc1}).expand({nc3, nc2, nc1}));
  vars["hydro_w"] = w;
}

}  // namespace

TEST(HydrostaticAtmosphere, cubed_sphere_remains_at_rest) {
  torch::set_num_threads(1);
  torch::set_num_interop_threads(1);

  auto fname = write_temp_config();
  auto mesh = Mesh(MeshOptionsImpl::from_yaml(fname));
  std::remove(fname.c_str());
  mesh->to(torch::kCPU, torch::kFloat64);
  ASSERT_EQ(mesh->blocks.size(), 6);

  MeshVariables vars(mesh->blocks.size());
  for (size_t i = 0; i < mesh->blocks.size(); ++i) {
    initialize_hydrostatic_column(mesh->blocks[i], vars[i]);
  }
  mesh->initialize(vars);

  constexpr int nsteps = 100;
  double max_vertical_velocity = 0.;
  for (int step = 0; step < nsteps; ++step) {
    auto dt = mesh->max_time_step(vars);
    for (int stage = 0; stage < mesh->blocks.front()->pintg->stages.size();
         ++stage) {
      mesh->forward(vars, dt, stage);
    }

    for (size_t i = 0; i < mesh->blocks.size(); ++i) {
      auto interior = mesh->blocks[i]->part(
          {0, 0, 0}, PartOptions().exterior(false).ndim(3));
      auto vertical_velocity =
          vars[i].at("hydro_w")[IVX].index(interior).abs().max().item<double>();
      max_vertical_velocity =
          std::max(max_vertical_velocity, vertical_velocity);
    }
  }

  std::cout << "maximum spurious vertical velocity after " << nsteps
            << " steps: " << max_vertical_velocity << std::endl;
  EXPECT_TRUE(std::isfinite(max_vertical_velocity));
  EXPECT_LT(max_vertical_velocity, 1.e-8);
}
