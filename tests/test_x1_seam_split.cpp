// One process, a spherical-polar column as one block and as two blocks split
// in x1, stepped together: the split column must keep the one-block state.
//
// SNAP_X1_CENTROID_EXACT reads past the seam twice: the r^2 -> plain-mean
// conversion of the ghost cells (five-cell window) and the pressure source
// (six-face window, two faces past the seam). Both take the neighbour's
// values (hydro.cpp _x1_ghost_rows), so after 20 steps of a seeded column
// the two states agree to round-off, with the full pressure force and in
// hydrostatic-split mode. RED before that exchange (one-sided windows at
// every block edge).
//
// SNAP_GRAVITY_WORK_RADIAL_EXACT keeps its slope one-sided at every block
// edge (its ghost density change is not exchanged), so a split column
// differs; with that switch set in the environment (1, or 0 for the control)
// the second test prints the gap at nz 32/64/128 and, on, checks E + P on the
// split column (docs/derivations/curved-gravity-work-weight.md sec 7).
//
// SNAP_WB_REF4 alone (ctest test_x1_seam_split_wb_ref4): each block computes
// the resolution flag from its own scan pressures, ghosts included, and only
// (pref, dref) are exchanged; on a column cold enough that the flag switches
// on one cell above the seam, the split column must still keep the one-block
// state (docs/derivations/wb-ref4.md sec 7).

// external
#include <gtest/gtest.h>

// C/C++
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

// torch
#include <torch/torch.h>
#include <unistd.h>

// snap
#include <snap/snap.h>

#include <snap/coord/x1_centroid.hpp>
#include <snap/hydro/gravity_work_radial.hpp>
#include <snap/hydro/hydro.hpp>
#include <snap/hydro/wb_ref4.hpp>
#include <snap/mesh/mesh.hpp>

using namespace snap;

namespace {

constexpr double kR0 = 5.0, kLz = 2.0, kGamma = 1.4, kSeed = 0.05;

std::string column_yaml(int nx1, double nh, char const* gw) {
  char buf[2048];
  std::snprintf(buf, sizeof(buf), R"(
geometry:
  type: spherical-polar
  bounds: {x1min: %g, x1max: %g, x2min: 1.5207963267948966, x2max: 1.6207963267948966, x3min: 0., x3max: 0.1}
  cells: {nx1: %d, nx2: 1, nx3: 1, nghost: 3}
dynamics:
  equation-of-state:
    type: ideal-gas
    gammad: 1.4
    weight: 8.31446
    density-floor: 1.e-12
    pressure-floor: 1.e-12
    temperature-floor: 1.e-12
    limiter: true
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
  const-gravity: {grav1: -1., non-hydrostatic: %g, gravity-work: %s}
boundary-condition:
  external:
    x1-inner: reflecting
    x1-outer: reflecting
    x2-inner: reflecting
    x2-outer: reflecting
    x3-inner: periodic
    x3-outer: periodic
)",
                kR0, kR0 + kLz, nx1, nh, gw);
  return buf;
}

Mesh make_column(int nb1, int nx1, double nh, char const* gw) {
  char fname[] = "/tmp/x1-seam-split-XXXXXX";
  int fd = mkstemp(fname);
  EXPECT_NE(fd, -1);
  if (fd != -1) close(fd);
  std::ofstream out(fname);
  out << column_yaml(nx1, nh, gw);
  out.close();
  auto block_opts = MeshBlockOptionsImpl::from_yaml(fname);
  std::remove(fname);
  if (nb1 > 1) {
    block_opts->layout()->type() = "cubed";
    block_opts->layout()->pz(nb1);
  }
  auto mesh_opts = MeshOptionsImpl::create();
  mesh_opts->block(block_opts);
  mesh_opts->blocks_per_process(nb1);
  auto mesh = Mesh(mesh_opts);
  mesh->to(torch::kCPU, torch::kFloat64);
  return mesh;
}

// isothermal column (g = 1, H = 1) with a density bump on the mid seam and a
// sin(pi z / Lz) radial wind, ghosts included; cold: T = H = 0.25 e^{1.1 - z}
// instead, so dz/H = 0.5 at z = 1.1 when nx1 = 16 (hydrostatic in plane
// parallel: ln p = -int dz / T = -4 (e^{z - 1.1} - e^{-1.1}))
void fill_column(Mesh mesh, MeshVariables& vars, bool cold = false) {
  for (size_t b = 0; b < mesh->blocks.size(); ++b) {
    auto coord = mesh->blocks[b]->pcoord;
    int nc1 = coord->options->nc1();
    auto z = coord->x1v - kR0;
    auto temp = cold ? 0.25 * torch::exp(1.1 - z) : torch::ones_like(z);
    auto p = cold ? torch::exp(-4. * (torch::exp(z - 1.1) - std::exp(-1.1)))
                  : torch::exp(-z);
    auto rho =
        p / temp * (1. + 0.02 * torch::exp(-((z - 0.5 * kLz) / 0.2).square()));
    auto in = torch::logical_and(z > 0., z < kLz);
    auto v1 =
        torch::where(in, kSeed * std::sqrt(kGamma) * torch::sin(M_PI * z / kLz),
                     torch::zeros_like(z));
    auto w = torch::zeros({mesh->blocks[b]->phydro->peos->nvar(),
                           coord->options->nc3(), coord->options->nc2(), nc1},
                          torch::kFloat64);
    w[IDN].copy_(rho.view({1, 1, nc1}));
    w[IPR].copy_(p.view({1, 1, nc1}));
    w[IVX].copy_(v1.view({1, 1, nc1}));
    vars[b]["hydro_w"] = w;
  }
}

void step(Mesh mesh, MeshVariables& vars, double dt) {
  int nstage = mesh->blocks.front()->pintg->stages.size();
  for (int stage = 0; stage < nstage; ++stage) mesh->forward(vars, dt, stage);
}

std::vector<size_t> by_x1(Mesh mesh) {
  std::vector<size_t> order(mesh->blocks.size());
  for (size_t b = 0; b < order.size(); ++b) order[b] = b;
  std::sort(order.begin(), order.end(), [&](size_t a, size_t b) {
    return mesh->blocks[a]->pcoord->options->x1min() <
           mesh->blocks[b]->pcoord->options->x1min();
  });
  return order;
}

torch::Tensor column_state(Mesh mesh, MeshVariables const& vars) {
  std::vector<torch::Tensor> parts;
  for (auto b : by_x1(mesh)) {
    auto coord = mesh->blocks[b]->pcoord;
    parts.push_back(
        vars[b].at("hydro_u").slice(-1, coord->il(), coord->iu() + 1));
  }
  return torch::cat(parts, -1);
}

// sum V [E + rho phi(x1v) - g1 <(x1 - x1v)^2> s[rho]] over the blocks: the
// E + P that SNAP_GRAVITY_WORK_RADIAL_EXACT conserves (meshblock.cpp pe=)
double energy_p(Mesh mesh, MeshVariables const& vars) {
  double total = 0.;
  for (size_t b = 0; b < mesh->blocks.size(); ++b) {
    auto coord = mesh->blocks[b]->pcoord;
    int is = coord->il(), ie = coord->iu() + 1;
    auto u = vars[b].at("hydro_u");
    auto rho = u[IDN];
    auto pe = rho * coord->x1v;
    pe.slice(-1, is, ie) -= corrected_pe_work(rho.slice(-1, is, ie), coord->x1f,
                                              coord->x1v, is, ie, -1., true);
    auto vol = coord->cell_volume();
    total += ((u[IPR] + pe) * vol).slice(-1, is, ie).sum().item<double>();
  }
  return total;
}

// max over rho, rho v1 and E of |split - one| / max|one| after nstep steps
double split_gap(int nx1, double nh, char const* gw, int nstep,
                 double* ep_drift = nullptr, bool cold = false) {
  auto one = make_column(1, nx1, nh, gw);
  auto two = make_column(2, nx1, nh, gw);
  EXPECT_EQ(one->blocks.size(), 1u);
  EXPECT_EQ(two->blocks.size(), 2u);
  MeshVariables v1(1), v2(2);
  fill_column(one, v1, cold);
  fill_column(two, v2, cold);
  one->initialize(v1);
  two->initialize(v2);
  double dt = 0.3 * (kLz / nx1) / std::sqrt(kGamma);
  double ep0 = energy_p(two, v2), ep_max = 0.;
  for (int n = 0; n < nstep; ++n) {
    step(one, v1, dt);
    step(two, v2, dt);
    ep_max = std::max(ep_max, std::abs(energy_p(two, v2) - ep0) / ep0);
  }
  if (ep_drift) *ep_drift = ep_max;
  auto a = column_state(one, v1), c = column_state(two, v2);
  EXPECT_TRUE(a.sizes() == c.sizes());
  double gap = 0.;
  for (int var : {IDN, IVX, IPR}) {
    double d = (a[var] - c[var]).abs().max().item<double>();
    gap = std::max(gap, d / a[var].abs().max().item<double>());
  }
  return gap;
}

}  // namespace

TEST(X1SeamSplit, centroid_exact_split_matches_one_block) {
  if (std::getenv("SNAP_GRAVITY_WORK_RADIAL_EXACT") ||
      std::getenv("SNAP_WB_REF4"))
    GTEST_SKIP() << "run without SNAP_GRAVITY_WORK_RADIAL_EXACT, SNAP_WB_REF4";
  torch::set_num_threads(1);
  setenv("SNAP_X1_CENTROID_EXACT", "1", 1);
  ASSERT_TRUE(x1_centroid_exact_enabled());
  for (double nh : {1., 0.}) {
    double gap = split_gap(32, nh, "cell", 20);
    std::printf("non-hydrostatic %g: 2 blocks vs 1, max rel gap %.3e\n", nh,
                gap);
    // round-off: 20 steps of a 32-cell column
    EXPECT_LE(gap, 1e-13) << "non-hydrostatic " << nh;
  }
}

TEST(X1SeamSplit, radial_exact_split_gap) {
  if (!std::getenv("SNAP_GRAVITY_WORK_RADIAL_EXACT"))
    GTEST_SKIP() << "set SNAP_GRAVITY_WORK_RADIAL_EXACT=1 (or 0) to measure";
  bool on = HydroImpl::gravity_work_radial_exact();
  torch::set_num_threads(1);
  for (int nx1 : {32, 64, 128}) {
    double drift = 0.;
    double gap = split_gap(nx1, 1., "face", 20 * nx1 / 32, &drift);
    std::printf(
        "switch %d, nz %d: 2 blocks vs 1, max rel gap %.3e; split E+P "
        "drift %.3e\n",
        on, nx1, gap, drift);
    if (on) EXPECT_LE(drift, 1e-13) << "nz " << nx1;
  }
}

TEST(X1SeamSplit, wb_ref4_flag_at_the_seam_split_matches_one_block) {
  if (!std::getenv("SNAP_WB_REF4") || std::getenv("SNAP_X1_CENTROID_EXACT"))
    GTEST_SKIP() << "set SNAP_WB_REF4=1 alone";
  ASSERT_TRUE(wb_ref4_enabled());
  ASSERT_FALSE(x1_centroid_exact_enabled());
  torch::set_num_threads(1);
  for (double nh : {1., 0.}) {
    double gap = split_gap(16, nh, "cell", 20, nullptr, /*cold=*/true);
    std::printf(
        "SNAP_WB_REF4, cold, non-hydrostatic %g: 2 blocks vs 1, max rel gap "
        "%.3e\n",
        nh, gap);
    EXPECT_LE(gap, 1e-13) << "non-hydrostatic " << nh;
  }
}
