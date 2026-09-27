// external
#include <gtest/gtest.h>

// torch
#include <torch/torch.h>

// snap
#include <snap/snap.h>

#include <snap/mesh/meshblock.hpp>

// tests
#include "device_testing.hpp"

using namespace snap;

// A cloud species no nucleation reaction produces (the card's `rain`, made by
// coagulation) has no parent vapor to borrow a deficit from, so the conserved
// limiter clamps it to zero, which creates the mass it clamps. A repair of an
// over-drained cell must keep the column's total mass and leave dry air alone.
TEST_P(DeviceTest, parentless_cloud_repair_keeps_the_column_mass) {
  auto options = MeshBlockOptionsImpl::from_yaml("test_parentless_cloud.yaml");
  auto block = std::make_shared<MeshBlockImpl>(options);
  block->to(device, dtype);
  ASSERT_EQ(block->phydro->peos->nvar(), ICY + 3);  // vapor, cloud, rain

  auto coord = block->pcoord;
  int il = coord->il(), iu = coord->iu();
  auto cons = torch::zeros({block->phydro->peos->nvar(), coord->options->nc3(),
                            coord->options->nc2(), coord->options->nc1()},
                           torch::device(device).dtype(dtype));
  cons[IDN].fill_(1.);
  cons[IPR].fill_(1.e8);
  cons[ICY].fill_(0.01);       // vapor
  cons[ICY + 1].fill_(0.001);  // cloud
  cons[ICY + 2].fill_(0.001);  // rain
  double deficit = 1.e-4;      // one cell over-drained by the settling flux
  cons[ICY + 2].select(-1, il + 3).fill_(-deficit);
  auto column_mass = [&] {
    auto c = cons.slice(-1, il, iu + 1).to(torch::kFloat64);
    return (c[IDN] + c.narrow(0, ICY, 3).sum(0)).sum().item<double>();
  };
  double before = column_mass();
  auto dry = cons[IDN].clone();

  block->phydro->peos->apply_conserved_limiter_(cons);

  double after = column_mass();
  double tol = (dtype == torch::kFloat64 ? 1.e-12 : 1.e-6) * before;
  EXPECT_GE(cons[ICY + 2].min().item<double>(), 0.);
  EXPECT_TRUE(torch::equal(cons[IDN], dry)) << "dry air was debited";
  EXPECT_NEAR(after, before, tol)
      << "the repair created " << (after - before) / deficit
      << " of the deficit";
}
