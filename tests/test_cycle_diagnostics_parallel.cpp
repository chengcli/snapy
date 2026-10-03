// C/C++
#include <string>
#include <utility>
#include <vector>

// external
#include <gtest/gtest.h>

// snap
#include <snap/mesh/mesh.hpp>

using namespace snap;

namespace {

double read_token(std::string const& out, std::string const& key, bool* found) {
  auto pos = out.find(key);
  if (found) *found = pos != std::string::npos;
  if (pos == std::string::npos) return 0.;
  return std::stod(out.substr(pos + key.size()));
}

}  // namespace

TEST(cycle_info, aggregates_two_local_blocks_across_two_processes) {
  auto block_opts =
      MeshBlockOptionsImpl::from_yaml("test_mesh_multi_block.yaml");

  auto gravity = ConstGravityOptionsImpl::create();
  gravity->grav1(-10.);
  block_opts->hydro()->grav() = gravity;

  auto implicit = ImplicitOptionsImpl::create();
  implicit->scheme(1);
  block_opts->hydro()->icorr() = implicit;

  auto mesh_opts = MeshOptionsImpl::create();
  mesh_opts->block(block_opts);
  mesh_opts->blocks_per_process(2);
  auto mesh = Mesh(mesh_opts);

  auto layout = mesh->blocks.front()->get_layout();
  ASSERT_TRUE(layout->has_process_group());
  ASSERT_EQ(layout->options->process_world_size(), 2);
  ASSERT_EQ(mesh->blocks.size(), 2);
  int process_rank = layout->options->process_rank();

  MeshVariables vars(mesh->blocks.size());
  double mass = 0.;
  double ke = 0.;
  double energy = 0.;
  double pe = 0.;
  double lim_cut = 0.;
  double lim_flux = 0.;
  double severe = 0.;

  // Each global extremum is in the second local block, on opposite processes.
  constexpr double theta_by_rank[] = {0.9, 0.7, 0.8, 0.4};
  constexpr double vic_by_rank[] = {0.1, 0.4, 0.2, 0.3};

  for (size_t i = 0; i < mesh->blocks.size(); ++i) {
    auto block = mesh->blocks[i];
    block->pintg->options->ncycle_out(1);

    int block_rank = block->options->layout()->rank();
    double rho = block_rank + 1.;
    auto coord = block->pcoord;
    auto u = torch::zeros({block->phydro->peos->nvar(), coord->options->nc3(),
                           coord->options->nc2(), coord->options->nc1()},
                          torch::kFloat64);
    u[IDN].fill_(rho);
    u[IVX].fill_(2. * rho);
    u[IPR].fill_(10. * rho);
    vars[i]["hydro_u"] = u;

    auto interior =
        block->part({0, 0, 0}, PartOptions().exterior(false).ndim(3));
    auto vol = coord->cell_volume();
    double block_volume = vol.index(interior).sum().item<double>();
    mass += rho * block_volume;
    ke += 2. * rho * block_volume;
    energy += 10. * rho * block_volume;
    pe += (rho * 10. * coord->x1v * vol).index(interior).sum().item<double>();

    double meter = block_rank + 1.;
    block->phydro->lim_cut().fill_(meter);
    block->phydro->lim_flux().fill_(2. * meter);
    block->phydro->positivity_severe().fill_(meter);
    block->phydro->positivity_min().fill_(theta_by_rank[block_rank]);
    EXPECT_TRUE(block->phydro->picorr);
    if (block->phydro->picorr)
      block->phydro->picorr->clamp_residual().fill_(vic_by_rank[block_rank]);
    lim_cut += meter;
    lim_flux += 2. * meter;
    severe += meter;
  }

  auto expected = torch::tensor(
      {mass, ke, energy, pe, lim_cut, lim_flux, severe}, torch::kFloat64);
  std::vector<torch::Tensor> expected_reduce = {expected};
  layout->comm->allreduce(expected_reduce, c10d::ReduceOp::SUM);
  expected = expected_reduce[0];

  testing::internal::CaptureStdout();
  mesh->print_cycle_info(vars, 0., 1.);
  auto out = testing::internal::GetCapturedStdout();

  if (process_rank != layout->options->process_root_rank()) return;

  for (auto const& item : {
           std::pair{" mass0=", expected[0].item<double>()},
           {" ke=", expected[1].item<double>()},
           {" energy=", expected[2].item<double>()},
           {" pe=", expected[3].item<double>()},
           {" limcut=",
            expected[4].item<double>() / expected[5].item<double>()},
           {" thetasevere=", expected[6].item<double>()},
           {" thetamin=", 0.4},
           {" vicclamp=", 0.4},
       }) {
    bool found = false;
    double value = read_token(out, item.first, &found);
    EXPECT_TRUE(found) << item.first << " missing from " << out;
    EXPECT_NEAR(value, item.second, 1.e-11) << item.first << out;
  }

  auto first = out.find("cycle=");
  ASSERT_NE(first, std::string::npos) << out;
  EXPECT_EQ(out.find("cycle=", first + 1), std::string::npos) << out;
}
