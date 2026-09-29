// gtest
#include <gtest/gtest.h>

// C/C++
#include <iostream>

// snap
#include <snap/snap.h>

#include <snap/mesh/mesh.hpp>

using namespace snap;

// test_parentless_cloud_nb1's split column with one block per process: the
// over-drained rain cell is in rank 0's block and all the positive rain in
// rank 1's, so the column must be gathered through the process group (#232).
TEST(Mesh, parentless_cloud_repair_keeps_the_mass_across_processes) {
  auto block_opts =
      MeshBlockOptionsImpl::from_yaml("test_parentless_cloud_nb1.yaml");
  block_opts->layout()->type() = "cubed";
  block_opts->layout()->pz(2);
  auto mesh_opts = MeshOptionsImpl::create();
  mesh_opts->block(block_opts);
  mesh_opts->blocks_per_process(1);

  auto mesh = Mesh(mesh_opts);
  mesh->to(torch::kCPU, torch::kFloat64);
  auto block = mesh->blocks.front();
  auto layout = block->get_layout();
  ASSERT_TRUE(layout->has_process_group() &&
              layout->options->process_world_size() == 2 &&
              mesh->blocks.size() == 1)
      << "the two halves of the column must be in two processes";
  int rank = layout->options->process_rank();

  auto coord = block->pcoord;
  int il = coord->il(), iu = coord->iu();
  double x1min = coord->options->x1min();
  double dx = (coord->options->x1max() - x1min) / (iu - il + 1);
  auto cons = torch::zeros({block->phydro->peos->nvar(), coord->options->nc3(),
                            coord->options->nc2(), coord->options->nc1()},
                           torch::dtype(torch::kFloat64));
  cons[IDN].fill_(1.);
  cons[IPR].fill_(1.e8);
  cons[ICY].fill_(0.01);       // vapor
  cons[ICY + 1].fill_(0.001);  // cloud
  double deficit = 1.e-4;
  for (int i = il; i <= iu; ++i) {
    int g = static_cast<int>(x1min + (i - il + 0.5) * dx);  // global cell
    double rain = g >= 8 ? 0.001 : (g == 3 ? -deficit : 0.);
    cons[ICY + 2].select(-1, i).fill_(rain);
  }

  // global {rain, total} mass, summed over both ranks
  auto global_mass = [&] {
    auto c = (cons * coord->cell_volume()).slice(-1, il, iu + 1);
    auto mass = torch::stack(
        {c[ICY + 2].sum(), (c[IDN] + c.narrow(0, ICY, 3).sum(0)).sum()});
    std::vector<torch::Tensor> values = {mass};
    layout->comm->allreduce(values, c10d::ReduceOp::SUM);
    return values[0];
  };
  auto before = global_mass();

  // what advance_local() step (4) does on every block
  block->phydro->peos->apply_conserved_limiter_(cons, /*whole_column=*/true);

  // EXPECT, not ASSERT: both ranks must reach the collective below
  EXPECT_GE(cons[ICY + 2].min().item<double>(), 0.) << "rank " << rank;
  auto after = global_mass();
  double rain0 = before[0].item<double>(), rain1 = after[0].item<double>();
  double mass0 = before[1].item<double>(), mass1 = after[1].item<double>();
  std::cout << "rank " << rank << ": rain mass " << rain0 << " -> " << rain1
            << " (" << (rain1 - rain0) / deficit << " of the deficit)"
            << std::endl;
  EXPECT_NEAR(rain1, rain0, 1.e-12 * rain0)
      << "rank " << rank << ": rain mass changed by " << rain1 - rain0 << " ("
      << (rain1 - rain0) / deficit << " of the deficit)";
  EXPECT_NEAR(mass1, mass0, 1.e-12 * mass0)
      << "rank " << rank << ": total mass changed by " << mass1 - mass0;
}
