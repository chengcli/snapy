// C/C++
#include <string>

// external
#include <gtest/gtest.h>

// torch
#include <torch/torch.h>

// snap
#include <snap/snap.h>

#include <snap/forcing/forcing.hpp>
#include <snap/hydro/hydro.hpp>
#include <snap/mesh/meshblock.hpp>

using namespace snap;

namespace {
//! the one card this binary loads: kintera's species table is process-global.
//! It settles `cloud` at const-vsed -2 and carries no forcing block.
constexpr char const* kCard = "test_gravity_sedimentation.yaml";

torch::Tensor make_primitive(std::shared_ptr<MeshBlockImpl> const& block) {
  auto coord = block->pcoord;
  auto w = torch::zeros({block->phydro->peos->nvar(), coord->options->nc3(),
                         coord->options->nc2(), coord->options->nc1()},
                        torch::kFloat64);
  w[IDN].fill_(1.);
  w[IPR].fill_(1.e5);
  w[ICY].fill_(0.01);
  w[ICY + 1].fill_(0.02);
  return w;
}
}  // namespace

// sedimentation reads grav1 in forward(); without const-gravity that was a null
// dereference on the first step, so the card is refused at setup instead
TEST(sedimentation, refuses_a_card_without_const_gravity) {
  auto options = MeshBlockOptionsImpl::from_yaml(kCard);
  ASSERT_TRUE(options->hydro()->sed());
  ASSERT_FALSE(options->hydro()->grav());
  try {
    std::make_shared<MeshBlockImpl>(options);
    ADD_FAILURE() << "sedimentation without const-gravity accepted";
  } catch (std::exception const& e) {
    EXPECT_NE(std::string(e.what()).find("set forcing/const-gravity"),
              std::string::npos)
        << e.what();
  }
}

// With the x1 flux off nothing rewrites _flux1, so a settling flux added into
// it grew by one more copy on every call. from_yaml zeroes grav1 under
// disable-flux-x1, which makes sedimentation return early, so only options
// built in code reach this.
TEST(sedimentation, is_skipped_when_the_x1_flux_is_off) {
  auto options = MeshBlockOptionsImpl::from_yaml(kCard);
  auto gravity = ConstGravityOptionsImpl::create();
  gravity->grav1(-1.);
  options->hydro()->grav() = gravity;
  options->hydro()->disable_flux_x1() = true;
  auto block = std::make_shared<MeshBlockImpl>(options);

  auto w = make_primitive(block);
  auto u = block->phydro->peos->compute("W->U", {w});
  Variables vars;
  vars["hydro_w"] = torch::empty_like(w);

  auto du1 = block->phydro->forward(0.1, u.clone(), vars).clone();
  auto du2 = block->phydro->forward(0.1, u.clone(), vars).clone();

  // no x1 flux of any kind: the cloud does not move
  EXPECT_EQ(du1[ICY + 1].abs().max().item<double>(), 0.);
  EXPECT_TRUE(torch::equal(du1, du2))
      << "max |du2 - du1| = " << (du2 - du1).abs().max().item<double>();
}
