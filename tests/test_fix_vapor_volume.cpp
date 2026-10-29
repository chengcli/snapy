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

// the value the unweighted repair leaves in each of the two donor cells: the
// deficit d over the donors' vapor q + q, accumulated top down in T, taken
// from each donor in proportion to its own vapor
template <typename T>
double unweighted_fill(double q, double d) {
  T deficit = T(0.);
  T donors = T(0.);
  deficit -= T(-d);
  donors += T(q);
  donors += T(q);
  T share = deficit / donors;
  return double(T(q) - T(q) * share);
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

      if (geometry == "cartesian") {  // uniform column: the exact values
        double fill = dtype == torch::kFloat64 ? unweighted_fill<double>(q, d)
                                               : unweighted_fill<float>(q, d);
        for (int i = il + 1; i <= il + 2; ++i) {
          // within 4 ulp: a device sums in its own order
          double got = cons[slot].select(-1, i).item<double>();
          if (dtype == torch::kFloat64) {
            EXPECT_DOUBLE_EQ(got, fill) << label << ", cell " << i - il;
          } else {
            EXPECT_FLOAT_EQ(float(got), float(fill))
                << label << ", cell " << i - il;
          }
        }
        EXPECT_EQ(cons[slot].select(-1, il + 3).item<double>(), 0.) << label;
      }
    }
  }
}
