// RED on main 5eeb9b6. A resting polytrope that is uniform in x2, with
// reflecting x1 walls, periodic x2 and isotropic viscosity, grows u2 in the
// wall-adjacent row at the x2 block edge. Nothing in the state varies in x2, so
// u2 must stay zero. One block is enough: its periodic x2 edge is a seam too.
// Viscosity alone does it; the same column with kappa_iso alone stays at rest.

// C/C++
#include <cmath>
#include <cstdio>
#include <fstream>
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

// Anders & Brown (2017) polytrope, eps = 1e-2, n_rho = 3, in units Rd = 1
// (weight = Rgas); 32 x 8 cells
const char* kConfig = R"(
geometry:
  type: cartesian
  bounds: {x1min: 0., x1max: 6.488906699059797, x2min: 0., x2max: 1.6222266747649493, x3min: 0., x3max: 1.}
  cells: {nx1: 32, nx2: 8, nx3: 1, nghost: 3}

distribute:
  layout: slab
  nb2: 1
  nb3: 1
  blocks_per_process: 1

dynamics:
  equation-of-state:
    type: ideal-gas
    gammad: 1.6666666666666667
    weight: 8.31446
    density-floor: 1.e-12
    pressure-floor: 1.e-12
    temperature-floor: 1.e-12
  reconstruct:
    vertical: {type: weno5, scale: true, shock: false}
    horizontal: {type: weno5, scale: true, shock: false}
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
  cfl: 0.4
  implicit-scheme: 0

forcing:
  const-gravity:
    grav1: -2.49
  diffusion:
    nu_iso: 0.0234
)";

Mesh make_mesh() {
  char fname[] = "/tmp/wb-wall-corner-XXXXXX";
  int fd = mkstemp(fname);
  EXPECT_NE(fd, -1);
  if (fd != -1) close(fd);
  std::ofstream out(fname);
  out << kConfig;
  out.close();
  auto mesh = Mesh(MeshOptionsImpl::from_yaml(fname));
  std::remove(fname);
  mesh->to(torch::kCPU, torch::kFloat64);
  return mesh;
}

}  // namespace

// T = 1 + Lz - z, rho = T^m, p = rho T, at rest: uniform in x2. Ten steps.
TEST(WallCorner, x2_uniform_rest_keeps_u2_zero) {
  torch::set_num_threads(1);
  auto mesh = make_mesh();
  ASSERT_EQ(mesh->blocks.size(), 1u);
  auto block = mesh->blocks[0];
  auto coord = block->pcoord;
  constexpr double Lz = 6.488906699059797, m = 1.49;

  MeshVariables vars(1);
  auto w = torch::zeros({block->phydro->peos->nvar(), coord->options->nc3(),
                         coord->options->nc2(), coord->options->nc1()},
                        torch::kFloat64);
  auto temp = (1. + Lz - coord->x1v).view({1, 1, -1});
  w[IDN] = temp.pow(m).expand_as(w[IDN]);
  w[IPR] = w[IDN] * temp;
  vars[0]["hydro_w"] = w;
  mesh->initialize(vars);

  int nstage = block->pintg->stages.size();
  for (int step = 0; step < 10; ++step) {
    auto dt = mesh->max_time_step(vars);
    for (int stage = 0; stage < nstage; ++stage) mesh->forward(vars, dt, stage);
  }

  auto in = block->part({0, 0, 0}, PartOptions().exterior(false).ndim(3));
  auto wf = vars[0].at("hydro_w");
  auto cs = (5. / 3. * wf[IPR].index(in) / wf[IDN].index(in)).sqrt();
  auto mach2 = (wf[IVY].index(in).abs() / cs).max().item<double>();
  auto mach = (wf.narrow(0, IVX, 3).square().sum(0).index(in).sqrt() / cs)
                  .max()
                  .item<double>();
  // RED: on 5eeb9b6 u2 is not zero in the wall row at the x2 edge
  EXPECT_LE(mach2, 1e-12) << "max |u2|/c_s " << mach2;
  EXPECT_LE(mach, 1e-12) << "max |v|/c_s " << mach;
}
