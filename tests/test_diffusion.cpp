// C/C++
#include <cmath>

// external
#include <gtest/gtest.h>
#include <yaml-cpp/yaml.h>

// kintera
#include <kintera/species.hpp>

// torch
#include <torch/torch.h>

// snap
#include <snap/snap.h>

#include <snap/bc/bc_func.hpp>
#include <snap/coord/cubed_sphere_utils.hpp>
#include <snap/forcing/forcing.hpp>
#include <snap/mesh/mesh.hpp>
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

void initialize_hyperdiffusion_species() {
  static bool initialized = []() {
    kintera::init_species_from_yaml("test_scalar_hyperdiffusion.yaml");
    return true;
  }();
  (void)initialized;
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

TEST(diffusion_options, reject_conduction_without_reference_specific_heat) {
  auto options = MeshBlockOptionsImpl::from_yaml("test_diffusion.yaml");
  options->hydro()->eos()->type() = "shallow-water";
  options->hydro()->diffusion()->nu_iso() = 0.;
  EXPECT_ANY_THROW(std::make_shared<MeshBlockImpl>(options));
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
  auto cv_ref = block->phydro->peos->species_cv_ref();
  EXPECT_NEAR(du[IPR][0][0][4].item<double>(), 0.05 * cv_ref, 1.e-3);
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

TEST(scalar_hyperdiffusion_options, parse_and_validate) {
  auto options = ScalarHyperdiffusionOptionsImpl::from_yaml(YAML::Load(
      "scalar-hyperdiffusion: {damping-time: 12., fields: [vel1, vapor]}"));
  ASSERT_TRUE(options);
  EXPECT_DOUBLE_EQ(options->damping_time(), 12.);
  EXPECT_EQ(options->fields(), (std::vector<std::string>{"vel1", "vapor"}));

  EXPECT_ANY_THROW(ScalarHyperdiffusionOptionsImpl::from_yaml(
      YAML::Load("scalar-hyperdiffusion: {damping-time: 0., fields: [vel1]}")));
  EXPECT_ANY_THROW(ScalarHyperdiffusionOptionsImpl::from_yaml(
      YAML::Load("scalar-hyperdiffusion: {damping-time: 1., fields: []}")));
  EXPECT_ANY_THROW(ScalarHyperdiffusionOptionsImpl::from_yaml(YAML::Load(
      "scalar-hyperdiffusion: {damping-time: 1., fields: [vel1, vel1]}")));
}

TEST_P(DeviceTest, cubed_sphere_scalar_laplacian_preserves_constants) {
  initialize_hyperdiffusion_species();
  auto options =
      MeshBlockOptionsImpl::from_yaml("test_scalar_hyperdiffusion.yaml");
  options->layout()->rank(0);
  options->layout()->world_size(6);
  options->layout()->blocks_per_process(6);
  auto block = std::make_shared<MeshBlockImpl>(options);
  block->to(device, dtype);

  auto coord = block->pcoord;
  auto shape = std::vector<int64_t>{
      3, coord->options->nc3(), coord->options->nc2(), coord->options->nc1()};
  auto scalar = torch::ones(shape, torch::device(device).dtype(dtype));
  scalar[1].fill_(2.5);
  scalar[2].fill_(-0.25);
  auto density = torch::ones(
      {coord->options->nc3(), coord->options->nc2(), coord->options->nc1()},
      torch::device(device).dtype(dtype));

  auto result = block->phydro->pscalar_hyperdiffusion->laplacian->forward(
      scalar, density);
  EXPECT_TRUE(
      torch::allclose(result, torch::zeros_like(result), 1.e-10, 1.e-10));

  auto panel_mesh = torch::meshgrid({coord->x3v, coord->x2v}, "ij");
  auto lonlat = cs_ab_to_lonlat(CS_FACE_NAMES[0], panel_mesh[1], panel_mesh[0]);
  auto mode = std::get<1>(lonlat).sin().to(device, dtype).unsqueeze(-1);
  scalar.zero_();
  scalar[0].copy_(mode);
  result = block->phydro->pscalar_hyperdiffusion->laplacian->forward(scalar,
                                                                     density);
  auto central = std::vector<torch::indexing::TensorIndex>{
      torch::indexing::Slice(coord->kl() + 1, coord->ku()),
      torch::indexing::Slice(coord->jl() + 1, coord->ju()),
      torch::indexing::Slice()};
  auto radius2 = coord->x1v.to(device, dtype).square().view({1, 1, -1});
  EXPECT_TRUE(torch::allclose((result[0] * radius2).index(central),
                              (-2. * mode).index(central), 0.15, 0.15));
  EXPECT_NEAR(
      block->phydro->pscalar_hyperdiffusion->max_time_step(torch::stack(
          {density, density, density, density, density, density, density})),
      20., 1.e-10);
}

TEST(scalar_hyperdiffusion_options, reject_unknown_or_dry_species) {
  initialize_hyperdiffusion_species();
  auto options =
      MeshBlockOptionsImpl::from_yaml("test_scalar_hyperdiffusion.yaml");
  options->layout()->rank(0);
  options->layout()->world_size(6);
  options->layout()->blocks_per_process(6);
  options->hydro()->scalar_hyperdiffusion()->fields() = {"unknown"};
  EXPECT_ANY_THROW(std::make_shared<MeshBlockImpl>(options));

  options = MeshBlockOptionsImpl::from_yaml("test_scalar_hyperdiffusion.yaml");
  options->layout()->rank(0);
  options->layout()->world_size(6);
  options->layout()->blocks_per_process(6);
  options->hydro()->scalar_hyperdiffusion()->fields() = {"dry"};
  EXPECT_ANY_THROW(std::make_shared<MeshBlockImpl>(options));
}

TEST(scalar_hyperdiffusion, exchanges_intermediate_on_six_local_panels) {
  initialize_hyperdiffusion_species();
  auto mesh_options = MeshOptionsImpl::create();
  mesh_options->block(
      MeshBlockOptionsImpl::from_yaml("test_scalar_hyperdiffusion.yaml"));
  mesh_options->blocks_per_process(6);
  auto mesh = Mesh(mesh_options);
  MeshVariables vars(mesh->blocks.size());

  for (int n = 0; n < mesh->blocks.size(); ++n) {
    auto block = mesh->blocks[n];
    auto coord = block->pcoord;
    auto w = torch::zeros({block->phydro->peos->nvar(), coord->options->nc3(),
                           coord->options->nc2(), coord->options->nc1()},
                          torch::kFloat64);
    w[IDN].fill_(1.);
    w[IPR].fill_(1.e-3);
    auto face = std::get<2>(
        block->get_layout()->loc_of(block->options->layout()->rank()));
    auto panel_mesh = torch::meshgrid({coord->x3v, coord->x2v}, "ij");
    auto lonlat =
        cs_ab_to_lonlat(CS_FACE_NAMES[face], panel_mesh[1], panel_mesh[0]);
    auto mode = std::get<1>(lonlat).sin().unsqueeze(-1);
    w[IVX] = 0.2 + 0.01 * mode;
    w[ICY] = 0.1 + 0.01 * mode;
    w[ICY + 1].fill_(0.05);
    vars[n]["hydro_w"] = w;
  }

  mesh->initialize(vars);
  std::vector<torch::Tensor> before;
  auto totals = [&](int id) {
    double total = 0.;
    for (int n = 0; n < mesh->blocks.size(); ++n) {
      auto block = mesh->blocks[n];
      auto interior =
          block->part({0, 0, 0}, PartOptions().exterior(false).ndim(3));
      total += (vars[n].at("hydro_u")[id].index(interior) *
                block->pcoord->cell_volume().index(interior))
                   .sum()
                   .item<double>();
    }
    return total;
  };
  auto momentum_before = totals(IVX);
  auto vapor_before = totals(ICY);
  auto cloud_before = totals(ICY + 1);
  auto energy_before = totals(IPR);
  for (auto const& block_vars : vars) {
    before.push_back(block_vars.at("hydro_u").clone());
  }
  mesh->forward(vars, 1.e-3, 0);
  bool changed = false;
  for (int n = 0; n < vars.size(); ++n) {
    auto const& block_vars = vars[n];
    EXPECT_TRUE(torch::isfinite(block_vars.at("hydro_u")).all().item<bool>());
    changed =
        changed || !torch::equal(before[n][IVX], block_vars.at("hydro_u")[IVX]);
  }
  EXPECT_TRUE(changed);
  EXPECT_NEAR(totals(IVX), momentum_before, 1.e-10 * std::abs(momentum_before));
  EXPECT_NEAR(totals(ICY), vapor_before, 1.e-10 * std::abs(vapor_before));
  EXPECT_NEAR(totals(ICY + 1), cloud_before, 1.e-10 * std::abs(cloud_before));
  EXPECT_NEAR(totals(IPR), energy_before, 1.e-8 * std::abs(energy_before));
}
