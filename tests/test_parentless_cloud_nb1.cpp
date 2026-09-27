// external
#include <gtest/gtest.h>

// torch
#include <torch/torch.h>

// snap
#include <snap/snap.h>

#include <snap/mesh/mesh.hpp>

// tests
#include "device_testing.hpp"

using namespace snap;

namespace {

// The same 16-cell global column as one meshblock (nb1 = 1) or split into nb1
// meshblocks along x1 (cubed layout; the slab layout rejects nb1 > 1). Rain (no
// nucleation parent, made by coagulation) is 0.001 in the upper half, zero in
// the lower half except one over-drained cell at -deficit. The global column
// total of rain is positive, so the repair must not need the clamp.
struct ColumnMass {
  double before, after;
};

ColumnMass repair_split_column(int nb1, torch::Device device,
                               torch::Dtype dtype, double deficit) {
  auto block_opts =
      MeshBlockOptionsImpl::from_yaml("test_parentless_cloud_nb1.yaml");
  if (nb1 > 1) {
    block_opts->layout()->type() = "cubed";
    block_opts->layout()->pz(nb1);
  }
  auto mesh_opts = MeshOptionsImpl::create();
  mesh_opts->block(block_opts);
  mesh_opts->blocks_per_process(nb1);

  auto mesh = Mesh(mesh_opts);
  mesh->to(device, dtype);
  EXPECT_EQ(static_cast<int>(mesh->blocks.size()), nb1);

  std::vector<torch::Tensor> cons(mesh->blocks.size());
  for (size_t b = 0; b < mesh->blocks.size(); ++b) {
    auto block = mesh->blocks[b];
    EXPECT_EQ(block->phydro->peos->nvar(), ICY + 3);  // vapor, cloud, rain
    auto coord = block->pcoord;
    int il = coord->il(), iu = coord->iu();
    double x1min = coord->options->x1min();
    double dx = (coord->options->x1max() - x1min) / (iu - il + 1);
    cons[b] = torch::zeros({block->phydro->peos->nvar(), coord->options->nc3(),
                            coord->options->nc2(), coord->options->nc1()},
                           torch::device(device).dtype(dtype));
    cons[b][IDN].fill_(1.);
    cons[b][IPR].fill_(1.e8);
    cons[b][ICY].fill_(0.01);       // vapor
    cons[b][ICY + 1].fill_(0.001);  // cloud
    for (int i = il; i <= iu; ++i) {
      int g = static_cast<int>(x1min + (i - il + 0.5) * dx);  // global cell
      double rain = g >= 8 ? 0.001 : (g == 3 ? -deficit : 0.);
      cons[b][ICY + 2].select(-1, i).fill_(rain);
    }
  }

  auto total_mass = [&] {
    double mass = 0.;
    for (size_t b = 0; b < mesh->blocks.size(); ++b) {
      int il = mesh->blocks[b]->pcoord->il(),
          iu = mesh->blocks[b]->pcoord->iu();
      auto c = cons[b].slice(-1, il, iu + 1).to(torch::kFloat64);
      mass += (c[IDN] + c.narrow(0, ICY, 3).sum(0)).sum().item<double>();
    }
    return mass;
  };
  ColumnMass m{total_mass(), 0.};

  // what advance_local() step (4) does on every block, before any exchange
  for (size_t b = 0; b < mesh->blocks.size(); ++b)
    mesh->blocks[b]->phydro->peos->apply_conserved_limiter_(cons[b]);

  for (size_t b = 0; b < mesh->blocks.size(); ++b)
    EXPECT_GE(cons[b][ICY + 2].min().item<double>(), 0.) << "block " << b;
  m.after = total_mass();
  return m;
}

}  // namespace

// One meshblock: the column sees its own positive rain above the over-drained
// cell and the repair keeps the total mass.
TEST_P(DeviceTest, parentless_cloud_repair_keeps_the_mass_with_nb1_1) {
  double deficit = 1.e-4;
  auto m = repair_split_column(1, device, dtype, deficit);
  double tol = (dtype == torch::kFloat64 ? 1.e-12 : 1.e-6) * m.before;
  EXPECT_NEAR(m.after, m.before, tol)
      << "nb1 = 1: total mass changed by " << m.after - m.before << " ("
      << (m.after - m.before) / deficit << " of the deficit)";
}

// Two meshblocks along x1: the over-drained cell sits in the lower block, all
// the positive rain in the upper one. The repair scans only the block-local
// column, gives up, and the clamp creates mass equal to the deficit.
TEST_P(DeviceTest, parentless_cloud_repair_keeps_the_mass_with_nb1_2) {
  double deficit = 1.e-4;
  auto m = repair_split_column(2, device, dtype, deficit);
  double tol = (dtype == torch::kFloat64 ? 1.e-12 : 1.e-6) * m.before;
  EXPECT_NEAR(m.after, m.before, tol)
      << "nb1 = 2: total mass changed by " << m.after - m.before << " ("
      << (m.after - m.before) / deficit << " of the deficit)";
}
