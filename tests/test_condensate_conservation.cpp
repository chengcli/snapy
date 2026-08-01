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

TEST_P(DeviceTest, multi_vapor_condensate_debits_stoichiometric_mass) {
  auto options =
      MeshBlockOptionsImpl::from_yaml("test_condensate_conservation.yaml");
  options->hydro()->eos()->limiter() = true;
  auto block = std::make_shared<MeshBlockImpl>(options);
  block->to(device, dtype);

  // The EOS must use the mapping cached during construction, not inspect the
  // reaction metadata each time the limiter runs.
  block->phydro->peos->options->thermo()->nucleation()->reactions().clear();

  auto coord = block->pcoord;
  auto cons = torch::zeros({block->phydro->peos->nvar(), coord->options->nc3(),
                            coord->options->nc2(), coord->options->nc1()},
                           torch::device(device).dtype(dtype));
  cons[IDN].fill_(1.);
  cons[IPR].fill_(1.e8);
  cons[ICY].fill_(0.4);        // NH3
  cons[ICY + 1].fill_(0.3);    // H2S
  cons[ICY + 2].fill_(-0.12);  // NH4SH
  auto total_before = cons[IDN] + cons.narrow(0, ICY, 3).sum(0);

  auto nh3_weight = block->phydro->peos->species_weight(1);
  auto h2s_weight = block->phydro->peos->species_weight(2);
  auto expected_nh3 = 0.4 - 0.12 * nh3_weight / (nh3_weight + h2s_weight);
  auto expected_h2s = 0.3 - 0.12 * h2s_weight / (nh3_weight + h2s_weight);

  block->phydro->peos->apply_conserved_limiter_(cons);

  auto total_after = cons[IDN] + cons.narrow(0, ICY, 3).sum(0);
  EXPECT_TRUE(torch::allclose(total_after, total_before, 1.e-12, 1.e-12));
  EXPECT_TRUE(torch::allclose(
      cons[ICY], torch::full_like(cons[ICY], expected_nh3), 1.e-6, 1.e-6));
  EXPECT_TRUE(torch::allclose(cons[ICY + 1],
                              torch::full_like(cons[ICY + 1], expected_h2s),
                              1.e-6, 1.e-6));
  EXPECT_TRUE(torch::equal(cons[ICY + 2], torch::zeros_like(cons[ICY + 2])));
}
