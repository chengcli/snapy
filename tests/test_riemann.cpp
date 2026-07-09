// C/C++
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <regex>
#include <sstream>

// external
#include <gtest/gtest.h>
#include <yaml-cpp/yaml.h>

// kintera
#include <kintera/species.hpp>

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

const char *small_ideal_gas_gravity_config = R"(
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

forcing:
  const-gravity:
    grav1: -1.0
    non-hydrostatic: 0.0

geometry:
  type: cartesian
  bounds: {x1min: 0., x1max: 1., x2min: 0., x2max: 1., x3min: 0., x3max: 1.}
  cells: {nx1: 10, nx2: 8, nx3: 1, nghost: 3}

boundary-condition:
  external:
    x1-inner: reflecting
    x1-outer: reflecting
    x2-inner: reflecting
    x2-outer: reflecting
    x3-inner: reflecting
    x3-outer: reflecting
)";

const char *small_ideal_gas_cubed_sphere_hllc_config = R"(
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
    type: hllc

distribute:
  layout: cubed-sphere
  blocks_per_process: 1
  verbose: false

geometry:
  type: gnomonic-equiangle
  bounds:
    x1min: 1.
    x1max: 2.
    x2min_pi: -0.25
    x2max_pi: 0.25
    x3min_pi: -0.25
    x3max_pi: 0.25
  cells: {nx1: 8, nx2: 6, nx3: 6, nghost: 3}

boundary-condition:
  external:
    x1-inner: reflecting
    x1-outer: reflecting
    x2-inner: custom
    x2-outer: custom
    x3-inner: custom
    x3-outer: custom
)";

const char *small_ideal_gas_implicit_config = R"(
reference-state:
  Tref: 300.
  Pref: 1.e5

species:
  - name: dry
    composition: {O: 0.42, N: 1.56, Ar: 0.01}
    cv_R: 2.5

integration:
  implicit-scheme: 1

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

forcing:
  const-gravity:
    grav1: -1.0
    non-hydrostatic: 0.0

geometry:
  type: cartesian
  bounds: {x1min: 0., x1max: 1., x2min: 0., x2max: 1., x3min: 0., x3max: 1.}
  cells: {nx1: 10, nx2: 8, nx3: 1, nghost: 3}

boundary-condition:
  external:
    x1-inner: reflecting
    x1-outer: reflecting
    x2-inner: reflecting
    x2-outer: reflecting
    x3-inner: reflecting
    x3-outer: reflecting
)";

const char *small_ideal_moist_sedimentation_config = R"(
reference-state:
  Tref: 300.
  Pref: 1.e5

species:
  - name: dry
    composition: {O: 0.42, N: 1.56, Ar: 0.01}
    cv_R: 2.5

  - name: vapor
    composition: {H: 2, O: 1}
    cv_R: 2.5
    u0_R: 0.

  - name: cloud
    composition: {H: 2, O: 1}
    cv_R: 9.0
    u0_R: -3430.

dynamics:
  equation-of-state:
    type: ideal-moist
    density-floor: 1.e-10
    pressure-floor: 1.e-10
    tracer-floor: 1.e-10
    limiter: false

  reconstruct:
    vertical: {type: weno5, scale: false, shock: false}
    horizontal: {type: weno5, scale: false, shock: false}

  riemann-solver:
    type: hllc

forcing:
  const-gravity:
    grav1: -9.8

sedimentation:
  radius:
    cloud: 1.0e-5

  density:
    cloud: 1000.

  const-vsed:
    cloud: -10.0

reactions:
  - equation: vapor <=> cloud
    type: nucleation
    rate-constant: {formula: h2o_ideal}

geometry:
  type: cartesian
  bounds: {x1min: 0., x1max: 1.e3, x2min: 0., x2max: 1.e3, x3min: 0., x3max: 1.e3}
  cells: {nx1: 10, nx2: 8, nx3: 1, nghost: 3}

boundary-condition:
  external:
    x1-inner: reflecting
    x1-outer: reflecting
    x2-inner: reflecting
    x2-outer: reflecting
    x3-inner: reflecting
    x3-outer: reflecting
)";

const char *small_ideal_moist_lmars_cloud_config = R"(
reference-state:
  Tref: 300.
  Pref: 1.e5

species:
  - name: dry
    composition: {O: 0.42, N: 1.56, Ar: 0.01}
    cv_R: 2.5

  - name: vapor
    composition: {H: 2, O: 1}
    cv_R: 2.5
    u0_R: 0.

  - name: cloud
    composition: {H: 2, O: 1}
    cv_R: 9.0
    u0_R: -3430.

dynamics:
  equation-of-state:
    type: ideal-moist
    density-floor: 1.e-10
    pressure-floor: 1.e-10
    limiter: true

  reconstruct:
    vertical: {type: weno5, scale: false, shock: false}
    horizontal: {type: weno5, scale: false, shock: false}

  riemann-solver:
    type: lmars

forcing:
  const-gravity:
    grav1: -9.8

reactions:
  - equation: vapor <=> cloud
    type: nucleation
    rate-constant: {formula: h2o_ideal}

geometry:
  type: cartesian
  bounds: {x1min: 0., x1max: 1.e3, x2min: 0., x2max: 1.e3, x3min: 0., x3max: 1.e3}
  cells: {nx1: 10, nx2: 8, nx3: 1, nghost: 3}

boundary-condition:
  external:
    x1-inner: reflecting
    x1-outer: reflecting
    x2-inner: reflecting
    x2-outer: reflecting
    x3-inner: reflecting
    x3-outer: reflecting
)";

const char *small_shallow_water_config = R"(
dynamics:
  equation-of-state:
    type: shallow-water
    density-floor: 1.e-10
    limiter: false

  reconstruct:
    vertical: {type: weno5, scale: false, shock: false}
    horizontal: {type: weno5, scale: false, shock: false}

  riemann-solver:
    type: shallow-roe
    dir: xy

geometry:
  type: cartesian
  bounds: {x1min: 0., x1max: 1., x2min: 0., x2max: 1., x3min: 0., x3max: 1.}
  cells: {nx1: 10, nx2: 8, nx3: 1, nghost: 3}

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

namespace {

struct ScopedEnv {
  explicit ScopedEnv(char const *name_) : name(name_) {
    auto *value = std::getenv(name);
    if (value) {
      had_value = true;
      old_value = value;
    }
  }
  ~ScopedEnv() {
    if (had_value) {
      setenv(name, old_value.c_str(), 1);
    } else {
      unsetenv(name);
    }
  }

  char const *name;
  bool had_value = false;
  std::string old_value;
};

}  // namespace

TEST(HydroOptions, auto_detects_supported_fused_recon_riemann) {
  char fname[80] = "/tmp/tempfile.XXXXXX";
  mkstemp(fname);
  std::ofstream outfile(fname);
  outfile << small_ideal_gas_config;
  outfile.close();

  auto op_block = MeshBlockOptionsImpl::from_yaml(fname);
  EXPECT_TRUE(op_block->hydro()->fused_recon_riemann());

  std::remove(fname);
}

TEST(HydroOptions, fused_env_disables_supported_fused_recon_riemann) {
  ScopedEnv env("FUSED");
  setenv("FUSED", "OFF", 1);

  char fname[80] = "/tmp/tempfile.XXXXXX";
  mkstemp(fname);
  std::ofstream outfile(fname);
  outfile << small_ideal_gas_config;
  outfile.close();

  auto op_block = MeshBlockOptionsImpl::from_yaml(fname);
  EXPECT_FALSE(op_block->hydro()->fused_recon_riemann());

  std::remove(fname);
}

TEST(HydroOptions, fused_env_auto_preserves_auto_detection) {
  ScopedEnv env("FUSED");
  setenv("FUSED", "AUTO", 1);

  char fname[80] = "/tmp/tempfile.XXXXXX";
  mkstemp(fname);
  std::ofstream outfile(fname);
  outfile << small_ideal_gas_config;
  outfile.close();

  auto op_block = MeshBlockOptionsImpl::from_yaml(fname);
  EXPECT_TRUE(op_block->hydro()->fused_recon_riemann());

  std::remove(fname);
}

TEST(HydroOptions, fused_env_auto_falls_back_for_unsupported_configuration) {
  ScopedEnv env("FUSED");
  setenv("FUSED", "AUTO", 1);

  char fname[80] = "/tmp/tempfile.XXXXXX";
  mkstemp(fname);
  std::ofstream outfile(fname);
  outfile << block_config;
  outfile.close();

  auto op_block = MeshBlockOptionsImpl::from_yaml(fname);
  EXPECT_FALSE(op_block->hydro()->fused_recon_riemann());

  std::remove(fname);
}

TEST(HydroOptions, fused_env_rejects_invalid_value) {
  ScopedEnv env("FUSED");
  setenv("FUSED", "maybe", 1);

  char fname[80] = "/tmp/tempfile.XXXXXX";
  mkstemp(fname);
  std::ofstream outfile(fname);
  outfile << small_ideal_gas_config;
  outfile.close();

  EXPECT_THROW((void)MeshBlockOptionsImpl::from_yaml(fname), c10::Error);

  std::remove(fname);
}

TEST(HydroOptions, fused_env_on_rejects_unsupported_configuration) {
  ScopedEnv env("FUSED");
  setenv("FUSED", "ON", 1);

  char fname[80] = "/tmp/tempfile.XXXXXX";
  mkstemp(fname);
  std::ofstream outfile(fname);
  outfile << block_config;
  outfile.close();

  EXPECT_THROW((void)MeshBlockOptionsImpl::from_yaml(fname), c10::Error);

  std::remove(fname);
}

TEST(HydroOptions, auto_detects_fused_recon_riemann_for_cartesian_multiblock) {
  auto config = std::string(
                    "distribute:\n"
                    "  layout: slab\n"
                    "  blocks_per_process: 3\n\n") +
                small_ideal_gas_config;

  char fname[80] = "/tmp/tempfile.XXXXXX";
  mkstemp(fname);
  std::ofstream outfile(fname);
  outfile << config;
  outfile.close();

  auto op_block = MeshBlockOptionsImpl::from_yaml(fname);
  EXPECT_TRUE(op_block->hydro()->fused_recon_riemann());

  std::remove(fname);
}

TEST(HydroOptions, auto_detects_fused_recon_riemann_for_cubed_sphere_bpp1) {
  auto config = std::string(
                    "distribute:\n"
                    "  layout: cubed-sphere\n"
                    "  blocks_per_process: 1\n\n") +
                small_ideal_gas_config;

  char fname[80] = "/tmp/tempfile.XXXXXX";
  mkstemp(fname);
  std::ofstream outfile(fname);
  outfile << config;
  outfile.close();

  auto op_block = MeshBlockOptionsImpl::from_yaml(fname);
  EXPECT_TRUE(op_block->hydro()->fused_recon_riemann());

  std::remove(fname);
}

TEST(HydroOptions,
     auto_detects_fused_recon_riemann_for_cubed_sphere_multiblock) {
  auto config = std::string(
                    "distribute:\n"
                    "  layout: cubed-sphere\n"
                    "  blocks_per_process: 6\n\n") +
                small_ideal_gas_config;

  char fname[80] = "/tmp/tempfile.XXXXXX";
  mkstemp(fname);
  std::ofstream outfile(fname);
  outfile << config;
  outfile.close();

  auto op_block = MeshBlockOptionsImpl::from_yaml(fname);
  EXPECT_TRUE(op_block->hydro()->fused_recon_riemann());

  std::remove(fname);
}

TEST(HydroOptions, leaves_unsupported_fused_recon_riemann_off) {
  auto config = std::regex_replace(
      block_config, std::regex("riemann-solver:\\n    type: lmars"),
      "fused-recon-riemann: true\n\n  riemann-solver:\n    type: lmars");

  char fname[80] = "/tmp/tempfile.XXXXXX";
  mkstemp(fname);
  std::ofstream outfile(fname);
  outfile << config;
  outfile.close();

  auto op_block = MeshBlockOptionsImpl::from_yaml(fname);
  EXPECT_FALSE(op_block->hydro()->fused_recon_riemann());

  std::remove(fname);
}

TEST(RiemannSolver, hllc_writes_face_pressure_output) {
  auto config = std::regex_replace(std::string(small_ideal_gas_config),
                                   std::regex("type: lmars"), "type: hllc");

  char fname[80] = "/tmp/tempfile.XXXXXX";
  mkstemp(fname);
  std::ofstream outfile(fname);
  outfile << config;
  outfile.close();

  auto opts = MeshBlockOptionsImpl::from_yaml(fname);
  auto block = MeshBlock(opts);
  block->to(torch::kCPU, torch::kDouble);
  std::remove(fname);

  auto pcoord = block->pcoord;
  int nc1 = pcoord->options->nc1();
  int nc2 = pcoord->options->nc2();
  int nc3 = pcoord->options->nc3();
  int nvar = block->phydro->peos->nvar();

  torch::manual_seed(77);
  auto wl = torch::rand({nvar, nc3, nc2, nc1}, torch::kFloat64);
  auto wr = torch::rand({nvar, nc3, nc2, nc1}, torch::kFloat64);
  wl[IDN].add_(1.0);
  wr[IDN].add_(1.0);
  wl[IPR].add_(1.0);
  wr[IPR].add_(1.0);
  wl.narrow(0, IVX, 3).sub_(0.5).mul_(0.1);
  wr.narrow(0, IVX, 3).sub_(0.5).mul_(0.1);

  auto gamma_l = block->phydro->peos->compute("W->A", {wl});
  auto gamma_r = block->phydro->peos->compute("W->A", {wr});
  auto cl = block->phydro->peos->compute("WA->L", {wl, gamma_l});
  auto cr = block->phydro->peos->compute("WA->L", {wr, gamma_r});

  auto rhoa = 0.5 * (wl[IDN] + wr[IDN]);
  auto ca = 0.5 * (cl + cr);
  auto pmid = 0.5 * (wl[IPR] + wr[IPR] + (wl[IVX] - wr[IVX]) * rhoa * ca);
  auto ql =
      torch::sqrt(1.0 + (gamma_l + 1) / (2 * gamma_l) * (pmid / wl[IPR] - 1.0));
  ql = torch::where(pmid <= wl[IPR], torch::ones_like(ql), ql);
  auto qr =
      torch::sqrt(1.0 + (gamma_r + 1) / (2 * gamma_r) * (pmid / wr[IPR] - 1.0));
  qr = torch::where(pmid <= wr[IPR], torch::ones_like(qr), qr);
  auto al = wl[IVX] - cl * ql;
  auto ar = wr[IVX] + cr * qr;
  auto vxl = wl[IVX] - al;
  auto vxr = wr[IVX] - ar;
  auto tl = wl[IPR] + vxl * wl[IDN] * wl[IVX];
  auto tr = wr[IPR] + vxr * wr[IDN] * wr[IVX];
  auto ml = wl[IDN] * vxl;
  auto mr = -(wr[IDN] * vxr);
  auto expected = ((ml * tr + mr * tl) / (ml + mr)).clamp_min(0.0);

  auto flx = torch::zeros_like(wl);
  auto face_pressure = torch::zeros({nc3, nc2, nc1}, torch::kFloat64);
  block->phydro->priemann->forward(wl, wr, DIM1, flx, face_pressure);

  EXPECT_TRUE(torch::allclose(face_pressure, expected, 1.e-10, 1.e-10));
}

TEST(RiemannSolver, lmars_writes_face_pressure_output) {
  char fname[80] = "/tmp/tempfile.XXXXXX";
  mkstemp(fname);
  std::ofstream outfile(fname);
  outfile << small_ideal_gas_config;
  outfile.close();

  auto opts = MeshBlockOptionsImpl::from_yaml(fname);
  auto block = MeshBlock(opts);
  block->to(torch::kCPU, torch::kDouble);
  std::remove(fname);

  auto pcoord = block->pcoord;
  int nc1 = pcoord->options->nc1();
  int nc2 = pcoord->options->nc2();
  int nc3 = pcoord->options->nc3();
  int nvar = block->phydro->peos->nvar();

  torch::manual_seed(78);
  auto wl = torch::rand({nvar, nc3, nc2, nc1}, torch::kFloat64);
  auto wr = torch::rand({nvar, nc3, nc2, nc1}, torch::kFloat64);
  wl[IDN].add_(1.0);
  wr[IDN].add_(1.0);
  wl[IPR].add_(1.0);
  wr[IPR].add_(1.0);
  wl.narrow(0, IVX, 3).sub_(0.5).mul_(0.1);
  wr.narrow(0, IVX, 3).sub_(0.5).mul_(0.1);

  auto gamma_l = block->phydro->peos->compute("W->A", {wl});
  auto gamma_r = block->phydro->peos->compute("W->A", {wr});
  auto rhobar = 0.5 * (wl[IDN] + wr[IDN]);
  auto gamma_bar = 0.5 * (gamma_l + gamma_r);
  auto cbar = torch::sqrt(gamma_bar * 0.5 * (wl[IPR] + wr[IPR]) / rhobar);
  auto expected =
      0.5 * (wl[IPR] + wr[IPR]) + 0.5 * (rhobar * cbar) * (wl[IVX] - wr[IVX]);

  auto flx = torch::zeros_like(wl);
  auto face_pressure = torch::zeros({nc3, nc2, nc1}, torch::kFloat64);
  block->phydro->priemann->forward(wl, wr, DIM1, flx, face_pressure);

  EXPECT_TRUE(torch::allclose(face_pressure, expected, 1.e-10, 1.e-10));
}

TEST(RiemannSolver, roe_writes_face_pressure_output) {
  auto config = std::regex_replace(std::string(small_ideal_gas_config),
                                   std::regex("type: lmars"), "type: roe");

  char fname[80] = "/tmp/tempfile.XXXXXX";
  mkstemp(fname);
  std::ofstream outfile(fname);
  outfile << config;
  outfile.close();

  auto opts = MeshBlockOptionsImpl::from_yaml(fname);
  auto block = MeshBlock(opts);
  block->to(torch::kCPU, torch::kDouble);
  std::remove(fname);

  auto pcoord = block->pcoord;
  int nc1 = pcoord->options->nc1();
  int nc2 = pcoord->options->nc2();
  int nc3 = pcoord->options->nc3();
  int nvar = block->phydro->peos->nvar();

  torch::manual_seed(79);
  auto wl = torch::rand({nvar, nc3, nc2, nc1}, torch::kFloat64);
  auto wr = torch::rand({nvar, nc3, nc2, nc1}, torch::kFloat64);
  wl[IDN].add_(1.0);
  wr[IDN].add_(1.0);
  wl[IPR].add_(1.0);
  wr[IPR].add_(1.0);
  wl.narrow(0, IVX, 3).sub_(0.5).mul_(0.1);
  wr.narrow(0, IVX, 3).sub_(0.5).mul_(0.1);

  auto ul = block->phydro->peos->compute("W->U", {wl});
  auto ur = block->phydro->peos->compute("W->U", {wr});
  auto sqrtdl = torch::sqrt(wl[IDN]);
  auto sqrtdr = torch::sqrt(wr[IDN]);
  auto expected =
      ((ul[IPR] + wl[IPR]) / sqrtdl + (ur[IPR] + wr[IPR]) / sqrtdr) /
      (sqrtdl + sqrtdr);

  auto flx = torch::zeros_like(wl);
  auto face_pressure = torch::zeros({nc3, nc2, nc1}, torch::kFloat64);
  block->phydro->priemann->forward(wl, wr, DIM1, flx, face_pressure);

  EXPECT_TRUE(torch::allclose(face_pressure, expected, 1.e-10, 1.e-10));
  EXPECT_GT(torch::abs(flx).sum().item<double>(), 0.0);
}

TEST(RiemannSolver, roe_writes_face_pressure_output_ideal_moist) {
  auto config =
      std::regex_replace(std::string(small_ideal_moist_lmars_cloud_config),
                         std::regex("type: lmars"), "type: roe");

  char fname[80] = "/tmp/tempfile.XXXXXX";
  mkstemp(fname);
  std::ofstream outfile(fname);
  outfile << config;
  outfile.close();

  kintera::init_species_from_yaml(fname);
  auto opts = MeshBlockOptionsImpl::from_yaml(fname);
  auto block = MeshBlock(opts);
  block->to(torch::kCPU, torch::kDouble);
  std::remove(fname);

  auto pcoord = block->pcoord;
  int nc1 = pcoord->options->nc1();
  int nc2 = pcoord->options->nc2();
  int nc3 = pcoord->options->nc3();
  int nvar = block->phydro->peos->nvar();

  torch::manual_seed(80);
  auto wl = torch::rand({nvar, nc3, nc2, nc1}, torch::kFloat64);
  auto wr = torch::rand({nvar, nc3, nc2, nc1}, torch::kFloat64);
  wl[IDN].mul_(0.1).add_(1.0);
  wr[IDN].mul_(0.1).add_(1.0);
  wl[IPR].mul_(1000.0).add_(1.e5);
  wr[IPR].mul_(1000.0).add_(1.e5);
  wl.narrow(0, IVX, 3).sub_(0.5).mul_(0.1);
  wr.narrow(0, IVX, 3).sub_(0.5).mul_(0.1);
  wl[ICY].mul_(1.e-3).add_(1.5e-2);
  wr[ICY].mul_(1.e-3).add_(1.5e-2);
  wl[ICY + 1].mul_(1.e-3).add_(4.e-3);
  wr[ICY + 1].mul_(1.e-3).add_(4.e-3);

  auto ul = block->phydro->peos->compute("W->U", {wl});
  auto ur = block->phydro->peos->compute("W->U", {wr});
  auto sqrtdl = torch::sqrt(wl[IDN]);
  auto sqrtdr = torch::sqrt(wr[IDN]);
  auto expected =
      ((ul[IPR] + wl[IPR]) / sqrtdl + (ur[IPR] + wr[IPR]) / sqrtdr) /
      (sqrtdl + sqrtdr);

  auto flx = torch::zeros_like(wl);
  auto face_pressure = torch::zeros({nc3, nc2, nc1}, torch::kFloat64);
  block->phydro->priemann->forward(wl, wr, DIM1, flx, face_pressure);

  EXPECT_TRUE(torch::allclose(face_pressure, expected, 1.e-10, 1.e-10));
  EXPECT_GT(torch::abs(flx).sum().item<double>(), 0.0);
}

TEST_P(DeviceTest, fused_recon_riemann_matches_staged_ideal_gas_roe) {
  if (device.type() != torch::kCUDA) {
    GTEST_SKIP() << "fused reconstruction/Riemann path is CUDA-only";
  }

  auto staged_config =
      std::regex_replace(std::string(small_ideal_gas_config),
                         std::regex("type: lmars"), "type: roe");
  char staged_name[80] = "/tmp/tempfile.XXXXXX";
  mkstemp(staged_name);
  std::ofstream staged_file(staged_name);
  staged_file << staged_config;
  staged_file.close();

  auto fused_config = std::regex_replace(
      staged_config, std::regex("riemann-solver:\\n    type: roe"),
      "fused-recon-riemann: true\n\n  riemann-solver:\n    type: roe");
  char fused_name[80] = "/tmp/tempfile.XXXXXX";
  mkstemp(fused_name);
  std::ofstream fused_file(fused_name);
  fused_file << fused_config;
  fused_file.close();

  auto staged_opts = MeshBlockOptionsImpl::from_yaml(staged_name);
  auto fused_opts = MeshBlockOptionsImpl::from_yaml(fused_name);
  staged_opts->hydro()->fused_recon_riemann() = false;
  auto staged_block = MeshBlock(staged_opts);
  auto fused_block = MeshBlock(fused_opts);
  staged_block->to(device, dtype);
  fused_block->to(device, dtype);

  int nc1 = staged_block->pcoord->options->nc1();
  int nc2 = staged_block->pcoord->options->nc2();
  int nc3 = staged_block->pcoord->options->nc3();
  int nvar = staged_block->phydro->peos->nvar();

  torch::manual_seed(205);
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
  EXPECT_TRUE(torch::allclose(fused_du, staged_du, 1.e-7, 1.e-7));

  std::remove(staged_name);
  std::remove(fused_name);
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
  staged_opts->hydro()->fused_recon_riemann() = false;
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

TEST_P(DeviceTest, fused_recon_riemann_matches_staged_ideal_gas_gravity) {
  if (device.type() != torch::kCUDA) {
    GTEST_SKIP() << "fused reconstruction/Riemann path is CUDA-only";
  }

  char staged_name[80] = "/tmp/tempfile.XXXXXX";
  mkstemp(staged_name);
  std::ofstream staged_file(staged_name);
  staged_file << small_ideal_gas_gravity_config;
  staged_file.close();

  char fused_name[80] = "/tmp/tempfile.XXXXXX";
  mkstemp(fused_name);
  std::ofstream fused_file(fused_name);
  fused_file << small_ideal_gas_gravity_config;
  fused_file.close();

  auto staged_opts = MeshBlockOptionsImpl::from_yaml(staged_name);
  auto fused_opts = MeshBlockOptionsImpl::from_yaml(fused_name);
  staged_opts->hydro()->fused_recon_riemann() = false;
  auto staged_block = MeshBlock(staged_opts);
  auto fused_block = MeshBlock(fused_opts);
  staged_block->to(device, dtype);
  fused_block->to(device, dtype);

  int nc1 = staged_block->pcoord->options->nc1();
  int nc2 = staged_block->pcoord->options->nc2();
  int nc3 = staged_block->pcoord->options->nc3();
  int nvar = staged_block->phydro->peos->nvar();

  torch::manual_seed(203);
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

TEST_P(DeviceTest, fused_recon_riemann_matches_staged_cubed_sphere_hllc) {
  if (device.type() != torch::kCUDA) {
    GTEST_SKIP() << "fused reconstruction/Riemann path is CUDA-only";
  }

  char staged_name[80] = "/tmp/tempfile.XXXXXX";
  mkstemp(staged_name);
  std::ofstream staged_file(staged_name);
  staged_file << small_ideal_gas_cubed_sphere_hllc_config;
  staged_file.close();

  auto fused_config = std::regex_replace(
      std::string(small_ideal_gas_cubed_sphere_hllc_config),
      std::regex("riemann-solver:\\n    type: hllc"),
      "fused-recon-riemann: true\n\n  riemann-solver:\n    type: hllc");
  char fused_name[80] = "/tmp/tempfile.XXXXXX";
  mkstemp(fused_name);
  std::ofstream fused_file(fused_name);
  fused_file << fused_config;
  fused_file.close();

  auto staged_opts = MeshBlockOptionsImpl::from_yaml(staged_name);
  auto fused_opts = MeshBlockOptionsImpl::from_yaml(fused_name);
  staged_opts->hydro()->fused_recon_riemann() = false;
  auto staged_block = MeshBlock(staged_opts);
  auto fused_block = MeshBlock(fused_opts);
  staged_block->to(device, dtype);
  fused_block->to(device, dtype);

  int nc1 = staged_block->pcoord->options->nc1();
  int nc2 = staged_block->pcoord->options->nc2();
  int nc3 = staged_block->pcoord->options->nc3();
  int nvar = staged_block->phydro->peos->nvar();

  torch::manual_seed(902);
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
  EXPECT_TRUE(torch::allclose(fused_du, staged_du, 1.e-7, 1.e-7));

  std::remove(staged_name);
  std::remove(fused_name);
}

TEST_P(DeviceTest, fused_recon_riemann_matches_staged_ideal_gas_implicit) {
  if (device.type() != torch::kCUDA) {
    GTEST_SKIP() << "fused reconstruction/Riemann path is CUDA-only";
  }

  char staged_name[80] = "/tmp/tempfile.XXXXXX";
  mkstemp(staged_name);
  std::ofstream staged_file(staged_name);
  staged_file << small_ideal_gas_implicit_config;
  staged_file.close();

  char fused_name[80] = "/tmp/tempfile.XXXXXX";
  mkstemp(fused_name);
  std::ofstream fused_file(fused_name);
  fused_file << small_ideal_gas_implicit_config;
  fused_file.close();

  auto staged_opts = MeshBlockOptionsImpl::from_yaml(staged_name);
  auto fused_opts = MeshBlockOptionsImpl::from_yaml(fused_name);
  staged_opts->hydro()->fused_recon_riemann() = false;
  auto staged_block = MeshBlock(staged_opts);
  auto fused_block = MeshBlock(fused_opts);
  staged_block->to(device, dtype);
  fused_block->to(device, dtype);

  int nc1 = staged_block->pcoord->options->nc1();
  int nc2 = staged_block->pcoord->options->nc2();
  int nc3 = staged_block->pcoord->options->nc3();
  int nvar = staged_block->phydro->peos->nvar();

  torch::manual_seed(204);
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
  EXPECT_TRUE(torch::allclose(fused_block->phydro->implicit_mass_correction(),
                              staged_block->phydro->implicit_mass_correction(),
                              1.e-8, 1.e-8));

  std::remove(staged_name);
  std::remove(fused_name);
}

TEST_P(DeviceTest,
       fused_recon_riemann_matches_staged_ideal_moist_sedimentation) {
  if (device.type() != torch::kCUDA) {
    GTEST_SKIP() << "fused reconstruction/Riemann path is CUDA-only";
  }

  char staged_name[80] = "/tmp/tempfile.XXXXXX";
  mkstemp(staged_name);
  std::ofstream staged_file(staged_name);
  staged_file << small_ideal_moist_sedimentation_config;
  staged_file.close();

  char fused_name[80] = "/tmp/tempfile.XXXXXX";
  mkstemp(fused_name);
  std::ofstream fused_file(fused_name);
  fused_file << small_ideal_moist_sedimentation_config;
  fused_file.close();

  kintera::init_species_from_yaml(staged_name);
  auto staged_opts = MeshBlockOptionsImpl::from_yaml(staged_name);
  kintera::init_species_from_yaml(fused_name);
  auto fused_opts = MeshBlockOptionsImpl::from_yaml(fused_name);
  staged_opts->hydro()->fused_recon_riemann() = false;
  staged_opts->hydro()->disable_flux_x1() = true;
  staged_opts->hydro()->disable_flux_x2() = true;
  staged_opts->hydro()->disable_flux_x3() = true;
  staged_opts->hydro()->grav()->grav1(-9.8);
  fused_opts->hydro()->disable_flux_x1() = true;
  fused_opts->hydro()->disable_flux_x2() = true;
  fused_opts->hydro()->disable_flux_x3() = true;
  fused_opts->hydro()->grav()->grav1(-9.8);
  auto staged_block = MeshBlock(staged_opts);
  auto fused_block = MeshBlock(fused_opts);
  staged_block->to(device, dtype);
  fused_block->to(device, dtype);

  int nc1 = staged_block->pcoord->options->nc1();
  int nc2 = staged_block->pcoord->options->nc2();
  int nc3 = staged_block->pcoord->options->nc3();
  int nvar = staged_block->phydro->peos->nvar();

  torch::manual_seed(205);
  auto w =
      torch::rand({nvar, nc3, nc2, nc1}, torch::device(device).dtype(dtype));
  w[IDN].mul_(0.1).add_(1.0);
  w[IPR].mul_(1000.0).add_(1.e5);
  w.narrow(0, IVX, 3).sub_(0.5).mul_(0.1);
  if (nvar > ICY) {
    w.narrow(0, ICY, nvar - ICY).mul_(1.e-3);
  }

  auto u = staged_block->phydro->peos->compute("W->U", {w});
  Variables staged_vars, fused_vars;
  staged_vars["hydro_w"] = torch::empty_like(w);
  fused_vars["hydro_w"] = torch::empty_like(w);

  auto staged_du = staged_block->phydro->forward(1.e-4, u.clone(), staged_vars);
  auto fused_du = fused_block->phydro->forward(1.e-4, u.clone(), fused_vars);
  double tol = dtype == torch::kFloat32 ? 1.e-4 : 1.e-8;
  EXPECT_TRUE(torch::allclose(fused_du, staged_du, tol, tol));

  std::remove(staged_name);
  std::remove(fused_name);
}

TEST_P(DeviceTest, fused_recon_riemann_matches_staged_ideal_moist_lmars_cloud) {
  if (device.type() != torch::kCUDA) {
    GTEST_SKIP() << "fused reconstruction/Riemann path is CUDA-only";
  }

  char staged_name[80] = "/tmp/tempfile.XXXXXX";
  mkstemp(staged_name);
  std::ofstream staged_file(staged_name);
  staged_file << small_ideal_moist_lmars_cloud_config;
  staged_file.close();

  char fused_name[80] = "/tmp/tempfile.XXXXXX";
  mkstemp(fused_name);
  std::ofstream fused_file(fused_name);
  fused_file << small_ideal_moist_lmars_cloud_config;
  fused_file.close();

  kintera::init_species_from_yaml(staged_name);
  auto staged_opts = MeshBlockOptionsImpl::from_yaml(staged_name);
  kintera::init_species_from_yaml(fused_name);
  auto fused_opts = MeshBlockOptionsImpl::from_yaml(fused_name);
  staged_opts->hydro()->fused_recon_riemann() = false;
  auto staged_block = MeshBlock(staged_opts);
  auto fused_block = MeshBlock(fused_opts);
  staged_block->to(device, dtype);
  fused_block->to(device, dtype);

  int nc1 = staged_block->pcoord->options->nc1();
  int nc2 = staged_block->pcoord->options->nc2();
  int nc3 = staged_block->pcoord->options->nc3();
  int nvar = staged_block->phydro->peos->nvar();

  torch::manual_seed(206);
  auto w =
      torch::rand({nvar, nc3, nc2, nc1}, torch::device(device).dtype(dtype));
  w[IDN].mul_(0.1).add_(1.0);
  w[IPR].mul_(1000.0).add_(1.e5);
  w.narrow(0, IVX, 3).sub_(0.5).mul_(0.1);
  w[ICY].mul_(1.e-3).add_(1.5e-2);
  w[ICY + 1].mul_(1.e-3).add_(4.e-3);

  auto u = staged_block->phydro->peos->compute("W->U", {w});
  Variables staged_vars, fused_vars;
  staged_vars["hydro_w"] = torch::empty_like(w);
  fused_vars["hydro_w"] = torch::empty_like(w);

  auto staged_du = staged_block->phydro->forward(1.e-4, u.clone(), staged_vars);
  auto fused_du = fused_block->phydro->forward(1.e-4, u.clone(), fused_vars);
  double tol = dtype == torch::kFloat32 ? 1.e-4 : 1.e-8;
  auto max_diff = (fused_du - staged_du).abs().max().item<double>();
  EXPECT_TRUE(torch::allclose(fused_du, staged_du, tol, tol))
      << "max diff = " << max_diff;

  std::remove(staged_name);
  std::remove(fused_name);
}

TEST_P(DeviceTest, fused_recon_riemann_matches_staged_ideal_moist_roe_cloud) {
  if (device.type() != torch::kCUDA) {
    GTEST_SKIP() << "fused reconstruction/Riemann path is CUDA-only";
  }

  auto staged_config =
      std::regex_replace(std::string(small_ideal_moist_lmars_cloud_config),
                         std::regex("type: lmars"), "type: roe");
  char staged_name[80] = "/tmp/tempfile.XXXXXX";
  mkstemp(staged_name);
  std::ofstream staged_file(staged_name);
  staged_file << staged_config;
  staged_file.close();

  auto fused_config = std::regex_replace(
      staged_config, std::regex("riemann-solver:\\n    type: roe"),
      "fused-recon-riemann: true\n\n  riemann-solver:\n    type: roe");
  char fused_name[80] = "/tmp/tempfile.XXXXXX";
  mkstemp(fused_name);
  std::ofstream fused_file(fused_name);
  fused_file << fused_config;
  fused_file.close();

  kintera::init_species_from_yaml(staged_name);
  auto staged_opts = MeshBlockOptionsImpl::from_yaml(staged_name);
  kintera::init_species_from_yaml(fused_name);
  auto fused_opts = MeshBlockOptionsImpl::from_yaml(fused_name);
  staged_opts->hydro()->fused_recon_riemann() = false;
  auto staged_block = MeshBlock(staged_opts);
  auto fused_block = MeshBlock(fused_opts);
  staged_block->to(device, dtype);
  fused_block->to(device, dtype);

  int nc1 = staged_block->pcoord->options->nc1();
  int nc2 = staged_block->pcoord->options->nc2();
  int nc3 = staged_block->pcoord->options->nc3();
  int nvar = staged_block->phydro->peos->nvar();

  torch::manual_seed(207);
  auto w =
      torch::rand({nvar, nc3, nc2, nc1}, torch::device(device).dtype(dtype));
  w[IDN].mul_(0.1).add_(1.0);
  w[IPR].mul_(1000.0).add_(1.e5);
  w.narrow(0, IVX, 3).sub_(0.5).mul_(0.1);
  w[ICY].mul_(1.e-3).add_(1.5e-2);
  w[ICY + 1].mul_(1.e-3).add_(4.e-3);

  auto u = staged_block->phydro->peos->compute("W->U", {w});
  Variables staged_vars, fused_vars;
  staged_vars["hydro_w"] = torch::empty_like(w);
  fused_vars["hydro_w"] = torch::empty_like(w);

  auto staged_du = staged_block->phydro->forward(1.e-4, u.clone(), staged_vars);
  auto fused_du = fused_block->phydro->forward(1.e-4, u.clone(), fused_vars);
  double tol = dtype == torch::kFloat32 ? 1.e-4 : 1.e-8;
  auto max_diff = (fused_du - staged_du).abs().max().item<double>();
  EXPECT_TRUE(torch::allclose(fused_du, staged_du, tol, tol))
      << "max diff = " << max_diff;

  std::remove(staged_name);
  std::remove(fused_name);
}

TEST_P(DeviceTest, fused_recon_riemann_matches_staged_shallow_water_roe) {
  if (device.type() != torch::kCUDA) {
    GTEST_SKIP() << "fused reconstruction/Riemann path is CUDA-only";
  }

  char staged_name[80] = "/tmp/tempfile.XXXXXX";
  mkstemp(staged_name);
  std::ofstream staged_file(staged_name);
  staged_file << small_shallow_water_config;
  staged_file.close();

  char fused_name[80] = "/tmp/tempfile.XXXXXX";
  mkstemp(fused_name);
  std::ofstream fused_file(fused_name);
  fused_file << small_shallow_water_config;
  fused_file.close();

  auto staged_opts = MeshBlockOptionsImpl::from_yaml(staged_name);
  auto fused_opts = MeshBlockOptionsImpl::from_yaml(fused_name);
  staged_opts->hydro()->fused_recon_riemann() = false;
  auto staged_block = MeshBlock(staged_opts);
  auto fused_block = MeshBlock(fused_opts);
  staged_block->to(device, dtype);
  fused_block->to(device, dtype);

  int nc1 = staged_block->pcoord->options->nc1();
  int nc2 = staged_block->pcoord->options->nc2();
  int nc3 = staged_block->pcoord->options->nc3();
  int nvar = staged_block->phydro->peos->nvar();

  torch::manual_seed(204);
  auto w =
      torch::rand({nvar, nc3, nc2, nc1}, torch::device(device).dtype(dtype));
  w[IDN].add_(1.0);
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
