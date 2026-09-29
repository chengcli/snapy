// C/C++
#include <cmath>
#include <iomanip>
#include <sstream>
#include <string>

// external
#include <gtest/gtest.h>
#include <yaml-cpp/yaml.h>

// torch
#include <torch/torch.h>

// snap
#include <snap/snap.h>

#include <snap/bc/bc_func.hpp>
#include <snap/forcing/forcing.hpp>
#include <snap/mesh/meshblock.hpp>

// tests
#include "device_testing.hpp"

using namespace snap;

// x1 profiles of the kinematic diffusion coefficients (nu_scale_x1,
// kappa_scale_x1): a tensor over the block's x1 cell centres, or a YAML table
// {x1: [...], scale: [...]} interpolated onto them.

namespace {

std::string table_yaml(char const* key, std::vector<double> const& x1,
                       std::vector<double> const& scale) {
  std::ostringstream os;
  os << std::setprecision(17) << key << ": {x1: [";
  for (size_t i = 0; i < x1.size(); ++i) os << (i ? ", " : "") << x1[i];
  os << "], scale: [";
  for (size_t i = 0; i < scale.size(); ++i) os << (i ? ", " : "") << scale[i];
  os << "]}";
  return os.str();
}

DiffusionOptions parse(std::string const& body) {
  return DiffusionOptionsImpl::from_yaml(
      YAML::Load("diffusion: {nu_iso: 0.5, kappa_iso: 0.25, " + body + "}"));
}

MeshBlockOptions base_options() {
  return MeshBlockOptionsImpl::from_yaml("test_diffusion.yaml");
}

std::shared_ptr<MeshBlockImpl> build(MeshBlockOptions const& options) {
  return std::make_shared<MeshBlockImpl>(options);
}

void make_x1_periodic(MeshBlockOptions const& options, int nx1) {
  options->coord()->global_nx1() = nx1;
  options->coord()->nx1() = nx1;
  options->coord()->global_x1max() = 2. * M_PI;
  options->coord()->x1max() = 2. * M_PI;
  options->bfuncs()[BoundaryFace::kInnerX1] =
      get_bc_func().at("periodic_inner");
  options->bfuncs()[BoundaryFace::kOuterX1] =
      get_bc_func().at("periodic_outer");
  options->bcnames()[BoundaryFace::kInnerX1] = "periodic_inner";
  options->bcnames()[BoundaryFace::kOuterX1] = "periodic_outer";
}

void fill_periodic_x1(torch::Tensor const& var, int nghost) {
  BoundaryFuncOptions options;
  options.type(kPrimitive).nghost(nghost);
  get_bc_func().at("periodic_inner")(var, 3, options);
  get_bc_func().at("periodic_outer")(var, 3, options);
}

//! rho linear, v2 = x^2, T = 300 + x^2: every term of both fluxes is non-zero
torch::Tensor sheared_state(std::shared_ptr<MeshBlockImpl> const& block,
                            torch::Device device, torch::Dtype dtype,
                            torch::Tensor* temp) {
  auto coord = block->pcoord;
  auto w = torch::zeros(
      {5, coord->options->nc3(), coord->options->nc2(), coord->options->nc1()},
      torch::device(device).dtype(dtype));
  auto x = coord->x1v.to(device, dtype).view({1, 1, -1});
  *temp = (300. + x.square()).expand_as(w[IDN]).clone();
  w[IDN] = 1. + 0.1 * x;
  w[IVY] = x.square();
  auto Rd = 8.31446261815324 / block->phydro->peos->options->weight();
  w[IPR] = w[IDN] * Rd * (*temp);
  return w;
}

torch::Tensor tendency(std::shared_ptr<MeshBlockImpl> const& block,
                       torch::Tensor const& w, torch::Tensor const& temp) {
  auto du = torch::zeros_like(w);
  block->phydro->pdiffusion->forward(du, w, temp, 0.1);
  return du;
}

std::vector<double> centres(MeshBlockOptions const& options) {
  auto x = build(options)->pcoord->x1v.to(torch::kFloat64).contiguous();
  return std::vector<double>(x.data_ptr<double>(),
                             x.data_ptr<double>() + x.numel());
}

}  // namespace

TEST(diffusion_x1_scale, yaml_table_parses) {
  auto op = parse(table_yaml("nu_scale_x1", {0., 3., 6.}, {1., 2., 4.}));
  ASSERT_TRUE(op->nu_scale_x1_table().defined());
  EXPECT_FALSE(op->kappa_scale_x1_table().defined());
  EXPECT_FALSE(op->nu_scale_x1().defined());
  EXPECT_TRUE(torch::equal(
      op->nu_scale_x1_table(),
      torch::tensor({0., 3., 6., 1., 2., 4.}, torch::kFloat64).view({2, 3})));
}

TEST(diffusion_x1_scale, yaml_table_rejects_bad_input) {
  for (char const* bad : {
           "nu_scale_x1: 2.0",                                // not a table
           "nu_scale_x1: {x1: [0., 6.]}",                     // no scale
           "nu_scale_x1: {x1: [0., 6.], scale: [1.]}",        // lengths
           "nu_scale_x1: {x1: [0.], scale: [1.]}",            // one point
           "nu_scale_x1: {x1: [6., 0.], scale: [1., 1.]}",    // decreasing
           "nu_scale_x1: {x1: [0., 0.], scale: [1., 1.]}",    // repeated
           "nu_scale_x1: {x1: [0., 6.], scale: [1., 0.]}",    // zero
           "nu_scale_x1: {x1: [0., 6.], scale: [1., -2.]}",   // negative
           "nu_scale_x1: {x1: [0., 6.], scale: [1., .nan]}",  // NaN
           "nu_scale_x1: {x1: [0., 6.], scale: [1., x]}",     // not a number
           "kappa_scale_x1: {x1: [0., 6.], scale: [1., 1.], z: [0.]}",
       }) {
    EXPECT_THROW(parse(bad), c10::Error) << bad;
  }
}

// a profile of ones takes the scaled path and must not change one bit
TEST_P(DeviceTest, unity_profile_is_bitwise_no_profile) {
  torch::Tensor temp;
  auto plain = build(base_options());
  plain->to(device, dtype);
  auto w = sheared_state(plain, device, dtype, &temp);
  auto expected = tendency(plain, w, temp);

  auto by_cells = base_options();
  auto nc1 = plain->pcoord->options->nc1();
  by_cells->hydro()->diffusion()->nu_scale_x1(
      torch::ones({nc1}, torch::kFloat64));
  by_cells->hydro()->diffusion()->kappa_scale_x1(
      torch::ones({nc1}, torch::kFloat64));
  auto block = build(by_cells);
  block->to(device, dtype);
  EXPECT_TRUE(torch::equal(tendency(block, w, temp), expected));

  auto by_table = base_options();
  by_table->hydro()->diffusion() =
      parse(table_yaml("nu_scale_x1", {0., 6.}, {1., 1.}) + ", " +
            table_yaml("kappa_scale_x1", {0., 6.}, {1., 1.}));
  block = build(by_table);
  block->to(device, dtype);
  EXPECT_TRUE(torch::equal(tendency(block, w, temp), expected));
}

// v2 = sin(x) on a periodic box, nu scaled by a uniform s: exp(-s nu k^2 t),
// k = 1. Same scheme and tolerance as test_diffusion's unscaled sine mode.
TEST_P(DeviceTest, viscous_sine_mode_decays_at_the_scaled_rate) {
  constexpr int nx1 = 64, nsteps = 100;
  constexpr double s = 2.5;
  auto options = base_options();
  make_x1_periodic(options, nx1);
  options->hydro()->diffusion() =
      parse(table_yaml("nu_scale_x1", {0., 2. * M_PI}, {s, s}));
  options->hydro()->diffusion()->kappa_iso(0.);
  auto block = build(options);
  block->to(device, dtype);
  auto coord = block->pcoord;
  auto w = torch::ones(
      {5, coord->options->nc3(), coord->options->nc2(), coord->options->nc1()},
      torch::device(device).dtype(dtype));
  w[IPR] = 1.e5;
  w.narrow(0, IVX, 3).zero_();
  auto x = coord->x1v.to(device, dtype).view({1, 1, -1});
  w[IVY] = torch::sin(x);
  fill_periodic_x1(w, coord->options->nghost());

  auto nu = block->phydro->pdiffusion->options->nu_iso();
  auto dx = 2. * M_PI / nx1;
  auto dt = 0.1 * dx * dx / (s * nu);
  for (int n = 0; n < nsteps; ++n) {
    auto temp = block->phydro->peos->compute("W->T", {w});
    auto du = torch::zeros_like(w);
    block->phydro->pdiffusion->forward(du, w, temp, dt);
    w[IVY] += du[IVY];  // rho = 1
    fill_periodic_x1(w, coord->options->nghost());
  }

  auto interior = block->part({0, 0, 0}, PartOptions().exterior(false).ndim(3));
  auto expected = torch::sin(x) * std::exp(-s * nu * nsteps * dt);
  EXPECT_TRUE(torch::allclose(w[IVY].index(interior), expected.index(interior),
                              3.e-4, 3.e-4));
}

// T = 300 + 100 sin(x) at rest, kappa scaled by a uniform s: the energy
// tendency is kappa s rho cv T'' and, at fixed rho, dT = dE / (rho cv), so the
// mode decays as exp(-s kappa k^2 t), k = 1.
TEST_P(DeviceTest, conductive_sine_mode_decays_at_the_scaled_rate) {
  constexpr int nx1 = 64, nsteps = 100;
  constexpr double s = 2.5, amp = 100.;
  auto options = base_options();
  make_x1_periodic(options, nx1);
  options->hydro()->diffusion() =
      parse(table_yaml("kappa_scale_x1", {0., 2. * M_PI}, {s, s}));
  options->hydro()->diffusion()->nu_iso(0.);
  auto block = build(options);
  block->to(device, dtype);
  auto coord = block->pcoord;
  auto peos = block->phydro->peos;
  auto w = torch::zeros(
      {5, coord->options->nc3(), coord->options->nc2(), coord->options->nc1()},
      torch::device(device).dtype(dtype));
  w[IDN] = 1.;
  auto x = coord->x1v.to(device, dtype).view({1, 1, -1});
  auto Rd = 8.31446261815324 / peos->options->weight();
  w[IPR] = Rd * (300. + amp * torch::sin(x));
  fill_periodic_x1(w, coord->options->nghost());

  auto kappa = block->phydro->pdiffusion->options->kappa_iso();
  auto dx = 2. * M_PI / nx1;
  auto dt = 0.1 * dx * dx / (s * kappa);
  for (int n = 0; n < nsteps; ++n) {
    auto temp = peos->compute("W->T", {w});
    auto du = torch::zeros_like(w);
    block->phydro->pdiffusion->forward(du, w, temp, dt);
    // p = rho Rd T and dT = dE / (rho cv): dp = (Rd / cv) dE
    w[IPR] += du[IPR] * Rd / peos->specific_heat_cv(w, temp);
    fill_periodic_x1(w, coord->options->nghost());
  }

  auto interior = block->part({0, 0, 0}, PartOptions().exterior(false).ndim(3));
  auto mode = (peos->compute("W->T", {w}) - 300.) / amp;
  auto expected = torch::sin(x) * std::exp(-s * kappa * nsteps * dt);
  EXPECT_TRUE(torch::allclose(mode.index(interior),
                              expected.expand_as(mode).index(interior), 3.e-4,
                              3.e-4));
}

// s(x) = 1 + x/10 at rho = 1 is the kinematic flux of rho = 1 + x/10 at s = 1:
// for v2 = x^2 (T uniform) and T = 300 + x^2 (at rest) the tendencies are
// dt nu (2 + 0.4 x) and dt kappa cv (2 + 0.4 x) in every cell, the wall cells
// included (the wall face extrapolates the linear product exactly), as
// test_diffusion's dynamic_coefficients_carry_no_density has for rho.
TEST(diffusion_x1_scale, linear_profile_gives_the_analytic_tendency) {
  auto options = base_options();
  auto x1v = centres(options);
  std::vector<double> ends = {1. + 0.1 * x1v.front(), 1. + 0.1 * x1v.back()};
  options->hydro()->diffusion() =
      parse(table_yaml("nu_scale_x1", {x1v.front(), x1v.back()}, ends) + ", " +
            table_yaml("kappa_scale_x1", {x1v.front(), x1v.back()}, ends));
  auto block = build(options);
  auto peos = block->phydro->peos;
  auto x = block->pcoord->x1v.to(torch::kFloat64).view({1, 1, -1});
  auto interior = block->part({0, 0, 0}, PartOptions().exterior(false).ndim(3));
  auto Rd = 8.31446261815324 / peos->options->weight();
  auto run = [&](bool heat, torch::Tensor* cv) {
    torch::Tensor unused;
    auto w = sheared_state(block, torch::kCPU, torch::kFloat64, &unused);
    w[IDN] = 1.;
    auto temp = (300. + (heat ? x.square() : 0. * x)).expand_as(w[IDN]);
    w[IPR] = w[IDN] * Rd * temp;
    w[IVY] = heat ? 0. * x : x.square();
    if (cv) *cv = peos->specific_heat_cv(w, temp).index(interior);
    return tendency(block, w, temp)[heat ? IPR : IVY].index(interior);
  };
  auto shape = (2. + 0.4 * x).expand({1, 1, x.size(2)}).index(interior);
  torch::Tensor cv;
  auto heat = run(true, &cv);
  auto shear = run(false, nullptr);
  EXPECT_TRUE(torch::allclose(shear, 0.05 * shape, 1.e-12, 0.)) << shear;
  EXPECT_TRUE(torch::allclose(heat, 0.025 * cv * shape, 1.e-12, 0.)) << heat;
}

// YAML knots on the cell centres give the profile set as a tensor, bit for bit
TEST_P(DeviceTest, yaml_table_equals_the_cells_profile) {
  auto x1v = centres(base_options());
  std::vector<double> values(x1v.size());
  for (size_t i = 0; i < x1v.size(); ++i)
    values[i] = 1. + 0.5 * std::cos(x1v[i]);

  auto by_cells = base_options();
  auto cells = torch::tensor(values, torch::kFloat64);
  by_cells->hydro()->diffusion()->nu_scale_x1(cells);
  by_cells->hydro()->diffusion()->kappa_scale_x1(cells.flip(0));
  auto a = build(by_cells);
  a->to(device, dtype);

  std::vector<double> reversed(values.rbegin(), values.rend());
  auto by_table = base_options();
  by_table->hydro()->diffusion() =
      parse(table_yaml("nu_scale_x1", x1v, values) + ", " +
            table_yaml("kappa_scale_x1", x1v, reversed));
  auto b = build(by_table);
  b->to(device, dtype);

  torch::Tensor temp;
  auto w = sheared_state(a, device, dtype, &temp);
  auto expected = tendency(a, w, temp);
  EXPECT_TRUE(torch::equal(tendency(b, w, temp), expected));
  auto plain = build(base_options());
  plain->to(device, dtype);
  EXPECT_FALSE(torch::equal(tendency(plain, w, temp), expected));
}

// dx = 1, one active dimension: dt = 1 / (2 max(nu max s_nu, kappa max s_k))
TEST(diffusion_x1_scale, timestep_uses_the_largest_scaled_coefficient) {
  auto w_of = [](std::shared_ptr<MeshBlockImpl> const& block) {
    torch::Tensor temp;
    return sheared_state(block, torch::kCPU, torch::kFloat64, &temp);
  };
  auto options = base_options();
  options->hydro()->diffusion() =
      parse(table_yaml("nu_scale_x1", {0., 6.}, {1., 4.}));
  auto block = build(options);
  EXPECT_NEAR(block->phydro->pdiffusion->max_time_step(w_of(block)),
              1. / (2. * 0.5 * 4.), 1.e-12);

  options = base_options();
  options->hydro()->diffusion() =
      parse(table_yaml("kappa_scale_x1", {0., 6.}, {1., 4.}));
  block = build(options);
  EXPECT_NEAR(block->phydro->pdiffusion->max_time_step(w_of(block)),
              1. / (2. * 0.25 * 4.), 1.e-12);
}

TEST(diffusion_x1_scale, reset_refuses_bad_profiles) {
  auto nc1 = static_cast<int64_t>(centres(base_options()).size());
  auto expect_refused = [](MeshBlockOptions const& options, char const* why) {
    EXPECT_THROW(build(options), c10::Error) << why;
  };

  auto options = base_options();
  options->hydro()->diffusion()->nu_scale_x1(
      torch::ones({nc1 + 1}, torch::kFloat64));
  expect_refused(options, "wrong length");

  options = base_options();
  auto bad = torch::ones({nc1}, torch::kFloat64);
  bad[0] = 0.;
  options->hydro()->diffusion()->kappa_scale_x1(bad);
  expect_refused(options, "non-positive");

  options = base_options();
  options->hydro()->diffusion() =
      parse(table_yaml("nu_scale_x1", {0., 6.}, {1., 1.}));
  options->hydro()->diffusion()->nu_scale_x1(
      torch::ones({nc1}, torch::kFloat64));
  expect_refused(options, "table and tensor both set");

  options = base_options();
  options->hydro()->diffusion() =
      parse(table_yaml("nu_scale_x1", {0., 6.}, {1., 1.}));
  options->hydro()->diffusion()->dynamic(true);
  expect_refused(options, "dynamic: true");

  // a profile set after the block is built never reached reset
  auto block = build(base_options());
  torch::Tensor temp;
  auto w = sheared_state(block, torch::kCPU, torch::kFloat64, &temp);
  block->phydro->pdiffusion->options->nu_scale_x1(
      torch::ones({nc1}, torch::kFloat64));
  EXPECT_THROW(tendency(block, w, temp), c10::Error);
}
