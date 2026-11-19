// V3 lane X, X4: the no-source baseline of X.rst section 5's chemistry-slot
// gate. One deck, test_parentless_cloud_nb1.yaml, as one block against two
// (the split of tests/test_parentless_cloud_nb1.cpp:33-45) through both
// branches of Mesh::forward; the EOS limiter set false before the Mesh is
// built; filled as test_parentless_cloud_nb1.cpp:57-66 with deficit 0. Every
// owned cell of hydro_u after each of 10 steps must match bitwise. (The deck
// has no scalars, so scalar_s is not compared.)

// external
#include <gtest/gtest.h>

// C/C++
#include <algorithm>
#include <vector>

// torch
#include <torch/torch.h>

// snap
#include <snap/snap.h>

#include <snap/mesh/mesh.hpp>

// tests
#include "device_testing.hpp"

using namespace snap;

namespace {

//! the owned hydro_u of the global column after each step
std::vector<torch::Tensor> run_column(int nb1, torch::Device device,
                                      torch::Dtype dtype, int nstep) {
  auto block_opts =
      MeshBlockOptionsImpl::from_yaml("test_parentless_cloud_nb1.yaml");
  block_opts->hydro()->eos()->limiter() = false;
  if (nb1 > 1) {
    block_opts->layout()->type() = "cubed";
    block_opts->layout()->pz(nb1);
  }
  auto mesh_opts = MeshOptionsImpl::create();
  mesh_opts->block(block_opts);
  mesh_opts->blocks_per_process(nb1);
  auto mesh = Mesh(mesh_opts);
  mesh->to(device, dtype);

  MeshVariables vars(mesh->blocks.size());
  for (size_t b = 0; b < mesh->blocks.size(); ++b) {
    auto block = mesh->blocks[b];
    auto coord = block->pcoord;
    int il = coord->il(), iu = coord->iu();
    double x1min = coord->options->x1min();
    double dx = (coord->options->x1max() - x1min) / (iu - il + 1);
    auto cons =
        torch::zeros({block->phydro->peos->nvar(), coord->options->nc3(),
                      coord->options->nc2(), coord->options->nc1()},
                     torch::device(device).dtype(dtype));
    cons[IDN].fill_(1.);
    cons[IPR].fill_(1.e8);
    cons[ICY].fill_(0.01);       // vapor
    cons[ICY + 1].fill_(0.001);  // cloud
    for (int i = 0; i < coord->options->nc1(); ++i) {
      int g = static_cast<int>(x1min + (i - il + 0.5) * dx);   // global cell
      cons[ICY + 2].select(-1, i).fill_(g >= 8 ? 0.001 : 0.);  // deficit 0
    }
    vars[b]["hydro_w"] = block->phydro->peos->compute("U->W", {cons});
  }
  mesh->initialize(vars);

  // blocks in x1 order, then their owned cells side by side
  std::vector<size_t> order(mesh->blocks.size());
  for (size_t b = 0; b < order.size(); ++b) order[b] = b;
  std::sort(order.begin(), order.end(), [&](size_t a, size_t b) {
    return mesh->blocks[a]->pcoord->options->x1min() <
           mesh->blocks[b]->pcoord->options->x1min();
  });
  auto owned = [&]() {
    std::vector<torch::Tensor> parts;
    for (auto b : order) {
      auto coord = mesh->blocks[b]->pcoord;
      parts.push_back(vars[b]
                          .at("hydro_u")
                          .slice(-1, coord->il(), coord->iu() + 1)
                          .cpu()
                          .clone());
    }
    return torch::cat(parts, -1);
  };

  std::vector<torch::Tensor> out;
  int nstage = static_cast<int>(mesh->blocks[0]->pintg->stages.size());
  for (int n = 0; n < nstep; ++n) {
    double dt = mesh->max_time_step(vars);
    for (int s = 0; s < nstage; ++s) mesh->forward(vars, dt, s);
    EXPECT_EQ(mesh->check_redo(vars), 0) << "nb1 = " << nb1 << " step " << n;
    out.push_back(owned());
  }
  return out;
}

}  // namespace

TEST_P(DeviceTest, chemistry_slot_no_source_baseline_one_block_equals_two) {
  auto one = run_column(1, device, dtype, 10);
  auto two = run_column(2, device, dtype, 10);
  ASSERT_EQ(one.size(), two.size());
  for (size_t n = 0; n < one.size(); ++n)
    EXPECT_TRUE(torch::equal(one[n], two[n]))
        << "step " << n << ": max |two - one| = "
        << (two[n].to(torch::kFloat64) - one[n].to(torch::kFloat64))
               .abs()
               .max()
               .item<double>();
}

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
