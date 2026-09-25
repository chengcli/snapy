// gtest
#include <gtest/gtest.h>

// base
#include <configure.h>

// snap
#include <snap/mesh/mesh.hpp>

using namespace snap;

// dt is one number for the whole run, so redo has to be one decision for the
// whole run. test_check_redo_floor.py drives six blocks in ONE process, where
// has_process_group() is false and the allreduce is skipped, so it covers the
// per-process half only; this is the cross-rank half.
TEST(Mesh, check_redo_is_one_decision_across_ranks) {
  auto block_opts =
      MeshBlockOptionsImpl::from_yaml("test_mesh_multi_block.yaml");
  auto mesh_opts = MeshOptionsImpl::create();
  mesh_opts->block(block_opts);
  mesh_opts->blocks_per_process(2);

  auto mesh = Mesh(mesh_opts);
  auto layout = mesh->blocks.front()->get_layout();
  ASSERT_TRUE(layout->has_process_group())
      << "one process only: check_redo skips its allreduce, so this test would "
         "gate nothing";
  int rank = layout->options->process_rank();

  MeshVariables vars(mesh->blocks.size());
  for (int i = 0; i < mesh->blocks.size(); ++i) {
    auto pcoord = mesh->blocks[i]->pcoord;
    auto hydro_w =
        torch::zeros({5, pcoord->options->nc3(), pcoord->options->nc2(),
                      pcoord->options->nc1()},
                     torch::TensorOptions().dtype(torch::kFloat64));
    hydro_w[IDN].fill_(1.0);
    hydro_w[IPR].fill_(1.0e5);
    vars[i]["hydro_w"] = hydro_w;
  }
  mesh->initialize(vars);

  // what stage 0 would have saved: check_redo restores from this buffer
  ASSERT_TRUE(mesh->blocks.front()->named_buffers().contains("u0"));
  std::vector<torch::Tensor> saved(mesh->blocks.size());
  for (int i = 0; i < mesh->blocks.size(); ++i) {
    mesh->blocks[i]->named_buffers()["u0"].copy_(vars[i].at("hydro_u"));
    saved[i] = vars[i].at("hydro_u").clone();
  }

  // control. EXPECT, not ASSERT: every rank must reach the collective below
  EXPECT_EQ(mesh->check_redo(vars), 0) << "rank " << rank;

  // one interior cell of one block of rank 0, put exactly at the density floor
  if (rank == 0) {
    auto u = vars[0].at("hydro_u");
    double rho_floor = mesh->blocks[0]->phydro->peos->options->density_floor();
    u[IDN][u.size(1) / 2][u.size(2) / 2][u.size(3) / 2].fill_(rho_floor);
  }

  EXPECT_EQ(mesh->check_redo(vars), 1)
      << "rank " << rank
      << " did not redo the step that rank 0 floored; the ranks now disagree "
         "about dt";
  for (int i = 0; i < mesh->blocks.size(); ++i) {
    EXPECT_TRUE(torch::equal(vars[i].at("hydro_u"), saved[i]))
        << "rank " << rank << " block " << i << " was not rolled back";
  }
}
