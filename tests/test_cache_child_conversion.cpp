// A conversion or a reset requested on the coordinate CHILD alone must reach
// the caches that the block's other modules (Hydro, ImplicitHydro, the layout)
// built from the coordinate's buffers.
//
// Each case warms a block (one HydroImpl::forward, which populates the grid
// caches), applies the operation to the coordinate child alone, and compares
// the du returned by the next HydroImpl::forward with the du of a fresh block
// that took the same operation before any call. Both blocks end with the same
// grid buffers and the same state, so any difference is a cache built from the
// old grid. A float64 base round-trips through float32 (the grid changes); a
// float32 base round-trips through float64, which is exact and is the control.
//
// The spherical-polar cases also run with SNAP_X1_CENTROID_EXACT=1
// (tests/CMakeLists.txt), which enables the #297 stencil caches that key on the
// device and dtype only.

// external
#include <gtest/gtest.h>

// C/C++
#include <cmath>
#include <cstdio>
#include <fstream>
#include <string>

// POSIX
#include <unistd.h>

// torch
#include <torch/torch.h>

// snap
#include <snap/snap.h>

#include <snap/coord/coordinate.hpp>
#include <snap/hydro/hydro.hpp>
#include <snap/mesh/mesh_functions.hpp>
#include <snap/mesh/meshblock.hpp>

#include "cuda_test_gate.hpp"

using namespace snap;

namespace {

// a 2-D implicit moist column; 7 x 9 cells over 1e4 x 2e4 m, so the cell
// widths are not exact in float32
const char* cartesian_implicit = R"(
reference-state:
  Tref: 0.
  Pref: 1.e5

species:
  - name: dry
    composition: {O: 0.42, N: 1.56, Ar: 0.01}
    cv_R: 2.5
  - name: H2O
    composition: {H: 2, O: 1}
    cv_R: 2.5
    u0_R: 0.
  - name: H2O(l)
    composition: {H: 2, O: 1}
    cv_R: 9.0
    u0_R: -3430.

reactions:
  - equation: H2O <=> H2O(l)
    type: nucleation
    rate-constant: {formula: h2o_ideal}

geometry:
  type: cartesian
  bounds: {x1min: 0., x1max: 10.e3, x2min: 0., x2max: 20.e3, x3min: -0.5, x3max: 0.5}
  cells: {nx1: 7, nx2: 9, nx3: 1, nghost: 3}

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

boundary-condition:
  external:
    x1-inner: reflecting
    x1-outer: reflecting
    x2-inner: periodic
    x2-outer: periodic
    x3-inner: periodic
    x3-outer: periodic

integration:
  type: rk3
  cfl: 0.9
  implicit-scheme: 1

forcing:
  const-gravity:
    grav1: -9.8
)";

// a dry spherical-polar shell, 7 x 6 x 5 cells
const char* spherical_polar = R"(
reference-state:
  Tref: 300.
  Pref: 1.e5

species:
  - name: dry
    composition: {H: 1.64813, He: 0.17591}
    cv_R: 2.5

geometry:
  type: spherical-polar
  bounds: {x1min: 6.0e6, x1max: 6.4e6, x2min: 1.2, x2max: 1.9, x3min: 0., x3max: 0.7}
  cells: {nx1: 7, nx2: 6, nx3: 5, nghost: 3}

dynamics:
  equation-of-state:
    type: ideal-gas
    density-floor: 1.e-20
    pressure-floor: 1.e-20
    limiter: false
  reconstruct:
    vertical: {type: weno5, scale: true, shock: false}
    horizontal: {type: weno5, scale: true, shock: false}
  riemann-solver:
    type: lmars

boundary-condition:
  external:
    x1-inner: reflecting
    x1-outer: reflecting
    x2-inner: reflecting
    x2-outer: reflecting
    x3-inner: periodic
    x3-outer: periodic

integration:
  type: rk3
  cfl: 0.5
  implicit-scheme: 0

forcing:
  const-gravity:
    grav1: -10.
)";

std::shared_ptr<MeshBlockImpl> make_block(char const* config,
                                          torch::Device device,
                                          torch::Dtype dtype) {
  char fname[] = "/tmp/test-cache-child-XXXXXX";
  int fd = mkstemp(fname);
  EXPECT_NE(fd, -1);
  if (fd != -1) close(fd);
  {
    std::ofstream out(fname);
    out << config;
  }
  auto block =
      std::make_shared<MeshBlockImpl>(MeshBlockOptionsImpl::from_yaml(fname));
  std::remove(fname);
  block->to(device, dtype);
  return block;
}

// a stratified, sheared state, the same function of the cell centres for
// every block
Variables make_state(std::shared_ptr<MeshBlockImpl> const& block,
                     torch::Device device, torch::Dtype dtype) {
  auto coord = block->pcoord;
  auto opts = torch::TensorOptions().dtype(dtype).device(device);
  int nvar = block->phydro->peos->nvar();
  auto z = coord->x1v.to(opts);
  auto x = coord->x2v.to(opts);
  auto z0 = z[0], dz = z[-1] - z[0];
  auto x0 = x[0], dxs = x[-1] - x[0];
  auto zz = ((z - z0) / dz).view({1, 1, -1});
  auto xx = ((x - x0) / dxs).view({1, -1, 1});
  auto w = torch::zeros({nvar, coord->options->nc3(), coord->options->nc2(),
                         coord->options->nc1()},
                        opts);
  w[IDN] = (1.2 * torch::exp(-0.5 * zz)).expand_as(w[IDN]);
  w[IPR] = (1.e5 * torch::exp(-0.5 * zz)).expand_as(w[IPR]);
  w[IVX] = (0.5 * torch::sin(2. * M_PI * xx) * torch::sin(M_PI * zz))
               .expand_as(w[IVX]);
  w[IVY] = (2. * torch::sin(M_PI * xx) * (2. * zz - 1.)).expand_as(w[IVY]);
  if (nvar > ICY) w[ICY] = 0.01;
  Variables vars;
  vars["hydro_w"] = w;
  block->initialize(vars);
  return vars;
}

torch::Tensor du_of(std::shared_ptr<MeshBlockImpl> const& block,
                    torch::Device device, torch::Dtype dtype) {
  auto vars = make_state(block, device, dtype);
  return block->phydro->forward(0.25, vars.at("hydro_u"), vars).clone();
}

std::shared_ptr<torch::nn::Module> coord_module(
    std::shared_ptr<MeshBlockImpl> const& block) {
  auto m = std::dynamic_pointer_cast<torch::nn::Module>(block->pcoord);
  EXPECT_NE(m, nullptr);
  return m;
}

void coord_round_trip(std::shared_ptr<MeshBlockImpl> const& block,
                      torch::Dtype dtype) {
  auto other = dtype == torch::kFloat64 ? torch::kFloat32 : torch::kFloat64;
  auto coord = coord_module(block);
  coord->to(other);
  coord->to(dtype);
}

// a stretched x1 grid: faces packed toward x1min
void coord_regrid(std::shared_ptr<MeshBlockImpl> const& block) {
  MeshGenerator stretched = [](torch::Tensor rx, float lo, float hi) {
    return lo + (hi - lo) * rx * rx * (2. - rx);
  };
  block->pcoord->reset_coordinates({stretched, nullptr, nullptr});
}

void expect_same_du(torch::Tensor const& warm, torch::Tensor const& fresh,
                    char const* what) {
  ASSERT_EQ(warm.sizes(), fresh.sizes()) << what;
  auto d = (warm - fresh).abs().to(torch::kFloat64);
  auto scale = fresh.abs().max().to(torch::kFloat64).item<double>();
  EXPECT_TRUE(torch::equal(warm, fresh))
      << what << ": warm du differs from a fresh block's, max|diff| = "
      << d.max().item<double>() << " (max|du| = " << scale << "), "
      << (warm != fresh).sum().item<int64_t>() << " of " << warm.numel()
      << " elements";
}

// warm, then the round trip of the coordinate child; fresh: round trip first
void round_trip_case(char const* config, torch::Device device,
                     torch::Dtype dtype, char const* what) {
  auto warm = make_block(config, device, dtype);
  du_of(warm, device, dtype);
  coord_round_trip(warm, dtype);
  auto got = du_of(warm, device, dtype);

  auto fresh = make_block(config, device, dtype);
  coord_round_trip(fresh, dtype);
  auto want = du_of(fresh, device, dtype);
  expect_same_du(got, want, what);
}

// warm, then reset_coordinates on the coordinate child; fresh: reset first
void regrid_case(char const* config, torch::Device device, torch::Dtype dtype,
                 char const* what) {
  auto warm = make_block(config, device, dtype);
  du_of(warm, device, dtype);
  coord_regrid(warm);
  auto got = du_of(warm, device, dtype);

  auto fresh = make_block(config, device, dtype);
  coord_regrid(fresh);
  auto want = du_of(fresh, device, dtype);
  expect_same_du(got, want, what);
}

#define SKIP_WITHOUT_CUDA()       \
  if (!snapy_cuda_test_enabled()) \
  GTEST_SKIP() << "snapy was not built with CUDA, or no device"

}  // namespace

// ---- Hydro + ImplicitHydro on an implicit Cartesian deck ----------------

TEST(CoordChildConversion, implicit_round_trip_cpu_f64) {
  round_trip_case(cartesian_implicit, torch::kCPU, torch::kFloat64,
                  "implicit, CPU, f64->f32->f64");
}
TEST(CoordChildConversion, implicit_round_trip_cpu_f32) {
  round_trip_case(cartesian_implicit, torch::kCPU, torch::kFloat32,
                  "implicit, CPU, f32->f64->f32");
}
TEST(CoordChildConversion, implicit_round_trip_cuda_f64) {
  SKIP_WITHOUT_CUDA();
  round_trip_case(cartesian_implicit, torch::kCUDA, torch::kFloat64,
                  "implicit, CUDA, f64->f32->f64");
}
TEST(CoordChildConversion, implicit_round_trip_cuda_f32) {
  SKIP_WITHOUT_CUDA();
  round_trip_case(cartesian_implicit, torch::kCUDA, torch::kFloat32,
                  "implicit, CUDA, f32->f64->f32");
}

TEST(CoordChildConversion, implicit_regrid_cpu_f64) {
  regrid_case(cartesian_implicit, torch::kCPU, torch::kFloat64,
              "implicit, CPU, f64, reset_coordinates");
}
TEST(CoordChildConversion, implicit_regrid_cuda_f64) {
  SKIP_WITHOUT_CUDA();
  regrid_case(cartesian_implicit, torch::kCUDA, torch::kFloat64,
              "implicit, CUDA, f64, reset_coordinates");
}

// ---- spherical-polar: the x1 stencil caches of #297 ---------------------

TEST(CoordChildConversion, spherical_round_trip_cpu_f64) {
  round_trip_case(spherical_polar, torch::kCPU, torch::kFloat64,
                  "spherical-polar, CPU, f64->f32->f64");
}
TEST(CoordChildConversion, spherical_round_trip_cuda_f64) {
  SKIP_WITHOUT_CUDA();
  round_trip_case(spherical_polar, torch::kCUDA, torch::kFloat64,
                  "spherical-polar, CUDA, f64->f32->f64");
}
TEST(CoordChildConversion, spherical_regrid_cpu_f64) {
  regrid_case(spherical_polar, torch::kCPU, torch::kFloat64,
              "spherical-polar, CPU, f64, reset_coordinates");
}
TEST(CoordChildConversion, spherical_regrid_cuda_f64) {
  SKIP_WITHOUT_CUDA();
  regrid_case(spherical_polar, torch::kCUDA, torch::kFloat64,
              "spherical-polar, CUDA, f64, reset_coordinates");
}
