// C/C++
#include <cstdio>

// external
#include <gtest/gtest.h>

// torch
#include <torch/torch.h>

// snap
#include <snap/snap.h>

#include <snap/hydro/hydro.hpp>
#include <snap/mesh/meshblock.hpp>

using namespace snap;

namespace {

//! WENO5 faces along x2 of a one-cell dry slot between two moist cells, the
//! shape of the TJ16 species trigger; `limiter` switches the EOS limiter
torch::Tensor slot_faces(bool limiter) {
  auto options = MeshBlockOptionsImpl::from_yaml("test_species_face_clamp.yaml");
  options->hydro()->eos()->limiter() = limiter;
  auto block = std::make_shared<MeshBlockImpl>(options);
  block->to(torch::kCPU, torch::kFloat64);
  auto coord = block->pcoord;
  EXPECT_TRUE(block->phydro->precon23->options->shock());

  auto w = torch::zeros({block->phydro->peos->nvar(), coord->options->nc3(),
                         coord->options->nc2(), coord->options->nc1()},
                        torch::kFloat64);
  w[IDN].fill_(0.8);
  w[IPR].fill_(1.e5);
  w[ICY].fill_(1.e-16);
  int slot = coord->jl() + 6;
  w[ICY].select(1, slot - 1).fill_(4.7e-3);
  w[ICY].select(1, slot).fill_(2.5e-9);
  w[ICY].select(1, slot + 1).fill_(1.9e-5);
  return block->phydro->precon23->forward(w, /*DIM2=*/2);
}

}  // namespace

// Under shock: true the hydro reconstruction returned before the species
// floor of the non-shock path, so a WENO5 undershoot left negative vapor
// faces. With the EOS limiter on they are floored at zero, and the faces that
// were not negative keep their values bit for bit.
TEST(recon, shock_species_faces_are_floored) {
  auto raw = slot_faces(/*limiter=*/false);
  auto floored = slot_faces(/*limiter=*/true);

  double raw_min = raw.narrow(1, ICY, 1).min().item<double>();
  double floored_min = floored.narrow(1, ICY, 1).min().item<double>();
  std::printf("vapor faces: min %.6g unfloored, %.6g with the limiter on\n",
              raw_min, floored_min);
  ASSERT_LT(raw_min, 0.);  // the slot does undershoot
  EXPECT_GE(floored_min, 0.);

  auto keep = raw.narrow(1, ICY, 1) >= 0.;
  EXPECT_TRUE(torch::equal(raw.narrow(1, ICY, 1).masked_select(keep),
                           floored.narrow(1, ICY, 1).masked_select(keep)));
  // the non-species rows are not touched
  EXPECT_TRUE(torch::equal(raw.narrow(1, 0, ICY), floored.narrow(1, 0, ICY)));
}
