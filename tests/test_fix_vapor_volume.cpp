// C/C++
#include <iostream>
#include <string>

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

namespace {

// the value the unweighted repair writes into each of the three cells: the
// column sum -d + q + q, accumulated top down in T, over a dry sum of 3
template <typename T>
double unweighted_fill(double q, double d) {
  T sum = T(0.);
  sum += T(-d);
  sum += T(q);
  sum += T(q);
  return double(sum / T(3.));
}

}  // namespace

// The column repair of a negative vapor (fix_vapor) must keep the column's
// vapor mass, sum(rho q V). On a grid whose x1 cell volume varies (here
// spherical-polar; gnomonic-equiangle in the GCM) a repair that keeps
// sum(rho q) instead moves mass (#241). A cloud with no parent vapor (`rain`)
// is repaired the same way. A uniform cartesian column is the control: the
// repair there must also keep its old values bit for bit.
TEST_P(DeviceTest, fix_vapor_keeps_the_column_mass_on_varying_cell_volume) {
  for (std::string geometry : {"spherical-polar", "cartesian"}) {
    for (int slot : {int(ICY), int(ICY) + 2}) {  // vapor, parentless rain
      auto options =
          MeshBlockOptionsImpl::from_yaml("test_fix_vapor_volume.yaml");
      options->coord()->type() = geometry;
      auto block = std::make_shared<MeshBlockImpl>(options);
      block->to(device, dtype);
      ASSERT_EQ(block->phydro->peos->nvar(), ICY + 3);  // vapor, cloud, rain

      auto coord = block->pcoord;
      int il = coord->il(), iu = coord->iu();
      auto cons =
          torch::zeros({block->phydro->peos->nvar(), coord->options->nc3(),
                        coord->options->nc2(), coord->options->nc1()},
                       torch::device(device).dtype(dtype));
      cons[IDN].fill_(1.);
      cons[IPR].fill_(1.e8);
      cons[ICY].fill_(0.01);       // vapor
      cons[ICY + 1].fill_(0.001);  // cloud
      cons[ICY + 2].fill_(0.001);  // rain
      double q = slot == ICY ? 0.01 : 0.001;
      double d = 1.5 * q;  // one negative cell, covered by the two below it
      cons[slot].select(-1, il + 3).fill_(-d);

      auto vol = coord->cell_volume().slice(-1, il, iu + 1).to(torch::kFloat64);
      auto column_mass = [&] {
        auto c = cons[slot].slice(-1, il, iu + 1).to(torch::kFloat64);
        return (c * vol).sum().item<double>();
      };
      double before = column_mass();

      block->phydro->peos->apply_conserved_limiter_(cons);

      double rel = (column_mass() - before) / before;
      std::string label =
          geometry + (slot == ICY ? ", vapor" : ", parentless rain");
      std::cout << label << ": relative change of sum(rho q V) = " << rel
                << std::endl;

      double tol = dtype == torch::kFloat64 ? 1.e-12 : 1.e-6;
      EXPECT_GE(cons[slot].min().item<double>(), 0.) << label;
      EXPECT_NEAR(rel, 0., tol) << label << ": the repair moved vapor mass";

      if (geometry == "cartesian") {  // uniform column: the old values
        double fill = dtype == torch::kFloat64 ? unweighted_fill<double>(q, d)
                                               : unweighted_fill<float>(q, d);
        for (int i = il + 1; i <= il + 3; ++i) {
          EXPECT_EQ(cons[slot].select(-1, i).item<double>(), fill)
              << label << ", cell " << i - il;
        }
      }
    }
  }
}
