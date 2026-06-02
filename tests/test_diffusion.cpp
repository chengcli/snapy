// C/C++
#include <cmath>

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

namespace {

std::shared_ptr<MeshBlockImpl> make_block() {
  return std::make_shared<MeshBlockImpl>(
      MeshBlockOptionsImpl::from_yaml("test_diffusion.yaml"));
}

std::shared_ptr<MeshBlockImpl> make_periodic_block(int nx1) {
  auto options = MeshBlockOptionsImpl::from_yaml("test_diffusion.yaml");
  options->coord()->global_nx1() = nx1;
  options->coord()->nx1() = nx1;
  options->coord()->global_x1max() = 2. * M_PI;
  options->coord()->x1max() = 2. * M_PI;
  options->hydro()->diffusion()->kappa_iso() = 0.;
  return std::make_shared<MeshBlockImpl>(options);
}

torch::Tensor make_primitive(std::shared_ptr<MeshBlockImpl> const& block,
                             torch::Device device, torch::Dtype dtype) {
  auto coord = block->pcoord;
  auto w = torch::ones(
      {5, coord->options->nc3(), coord->options->nc2(), coord->options->nc1()},
      torch::device(device).dtype(dtype));
  w[IPR] = 1.e5;
  w.narrow(0, IVX, 3).zero_();
  return w;
}

void fill_periodic_x1(torch::Tensor const& var, int nghost) {
  BoundaryFuncOptions options;
  options.type(kPrimitive).nghost(nghost);
  get_bc_func().at("periodic_inner")(var, 3, options);
  get_bc_func().at("periodic_outer")(var, 3, options);
}

}  // namespace

TEST(diffusion_options, parse_and_reject_legacy_keys) {
  auto options = DiffusionOptionsImpl::from_yaml(
      YAML::Load("diffusion: {nu_iso: 2.0, kappa_iso: 3.0}"));
  ASSERT_TRUE(options);
  EXPECT_DOUBLE_EQ(options->nu_iso(), 2.);
  EXPECT_DOUBLE_EQ(options->kappa_iso(), 3.);

  EXPECT_ANY_THROW(DiffusionOptionsImpl::from_yaml(
      YAML::Load("diffusion: {K: 2.0, type: theta}")));
  EXPECT_ANY_THROW(
      DiffusionOptionsImpl::from_yaml(YAML::Load("diffusion: {nu_iso: -1.0}")));
}

TEST(diffusion_options, reject_enabled_curved_coordinates) {
  auto options = MeshBlockOptionsImpl::from_yaml("test_diffusion.yaml");
  options->coord()->type() = "spherical-polar";
  EXPECT_ANY_THROW(std::make_shared<MeshBlockImpl>(options));

  options->hydro()->diffusion()->nu_iso() = 0.;
  options->hydro()->diffusion()->kappa_iso() = 0.;
  EXPECT_NO_THROW(std::make_shared<MeshBlockImpl>(options));
}

TEST_P(DeviceTest, uniform_state_has_zero_tendency) {
  auto block = make_block();
  block->to(device, dtype);
  EXPECT_TRUE(
      torch::allclose(block->pcoord->center_distance1(),
                      torch::ones_like(block->pcoord->center_distance1())));
  auto w = make_primitive(block, device, dtype);
  auto temp = block->phydro->peos->compute("W->T", {w});
  auto du = torch::zeros_like(w);

  block->phydro->pdiffusion->forward(du, w, temp, 0.1);
  EXPECT_TRUE(torch::allclose(du, torch::zeros_like(du), 1.e-6, 1.e-6));
}

TEST_P(DeviceTest, transverse_velocity_uses_viscous_laplacian) {
  auto block = make_block();
  block->to(device, dtype);
  auto w = make_primitive(block, device, dtype);
  auto x = block->pcoord->x1v.to(device, dtype);
  w[IVY] = x.square();
  auto temp = block->phydro->peos->compute("W->T", {w});
  auto du = torch::zeros_like(w);

  block->phydro->pdiffusion->forward(du, w, temp, 0.1);
  auto expected = torch::zeros_like(du[IVY]);
  expected.index(block->part({0, 0, 0}, PartOptions().exterior(false).ndim(3)))
      .fill_(0.1);
  EXPECT_TRUE(torch::allclose(du[IVY], expected, 1.e-5, 1.e-5));
}

TEST_P(DeviceTest, temperature_uses_conductive_laplacian) {
  auto block = make_block();
  block->to(device, dtype);
  auto w = make_primitive(block, device, dtype);
  auto x = block->pcoord->x1v.to(device, dtype);
  auto temp = x.square().view({1, 1, -1});
  auto Rd = 8.31446261815324 / block->phydro->peos->options->weight();
  w[IPR] = w[IDN] * Rd * temp;
  auto du = torch::zeros_like(w);

  block->phydro->pdiffusion->forward(du, w, temp, 0.1);
  EXPECT_NEAR(du[IPR][0][0][4].item<double>(), 0.05, 1.e-5);
  EXPECT_TRUE(torch::allclose(du[IDN], torch::zeros_like(du[IDN])));
  EXPECT_TRUE(torch::allclose(du.narrow(0, IVX, 3),
                              torch::zeros_like(du.narrow(0, IVX, 3))));
}

TEST_P(DeviceTest, viscous_sine_mode_matches_analytic_decay) {
  constexpr int nx1 = 64;
  constexpr int nsteps = 100;
  auto block = make_periodic_block(nx1);
  block->to(device, dtype);
  auto coord = block->pcoord;
  auto w = make_primitive(block, device, dtype);
  auto x = coord->x1v.to(device, dtype).view({1, 1, -1});
  w[IVY] = torch::sin(x);
  fill_periodic_x1(w, coord->options->nghost());

  auto nu = block->phydro->pdiffusion->options->nu_iso();
  auto dx = (coord->options->x1max() - coord->options->x1min()) / nx1;
  auto dt = 0.1 * dx * dx / nu;
  for (int n = 0; n < nsteps; ++n) {
    auto temp = block->phydro->peos->compute("W->T", {w});
    auto du = torch::zeros_like(w);
    block->phydro->pdiffusion->forward(du, w, temp, dt);
    w[IVY] += du[IVY];
    fill_periodic_x1(w, coord->options->nghost());
  }

  auto interior = block->part({0, 0, 0}, PartOptions().exterior(false).ndim(3));
  auto time = nsteps * dt;
  auto expected = torch::sin(x) * std::exp(-nu * time);
  EXPECT_TRUE(torch::allclose(w[IVY].index(interior), expected.index(interior),
                              3.e-4, 3.e-4));
}

TEST_P(DeviceTest, timestep_uses_largest_diffusivity) {
  auto block = make_block();
  block->to(device, dtype);
  auto w = make_primitive(block, device, dtype);
  EXPECT_NEAR(block->phydro->pdiffusion->max_time_step(w), 1., 1.e-12);

  w[IPR] = 1.e-6;
  EXPECT_NEAR(block->phydro->max_time_step(w), 1., 1.e-6);
}
