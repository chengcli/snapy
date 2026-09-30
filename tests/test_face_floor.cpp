// C/C++
#include <cmath>

// external
#include <gtest/gtest.h>

// torch
#include <torch/torch.h>

// snap
#include <snap/snap.h>

#include <snap/forcing/forcing.hpp>
#include <snap/mesh/meshblock.hpp>

using namespace snap;

namespace {

// Reflecting column, WENO5, gravity on, implicit correction off.
// One upper cell is dipped (1e-4) and the cell below it is cut to 0.2, so the
// well-balanced reconstruction overshoots non-positive and the positivity
// floor fires. When the floor writes the adjacent cell's density, the mass
// flux at that face is -6.3e-11. Writing the density reference dsf instead,
// with that same reference, makes it -9.5e-8. When the reference is still the
// bottom-anchored isentrope, it is -1.3e-6.
std::shared_ptr<MeshBlockImpl> make_block(torch::Device device) {
  auto options = MeshBlockOptionsImpl::from_yaml("test_face_floor.yaml");
  auto gravity = ConstGravityOptionsImpl::create();
  gravity->grav1(-10.);
  options->hydro()->grav() = gravity;
  options->hydro()->icorr() = nullptr;
  auto block = std::make_shared<MeshBlockImpl>(options);
  block->to(device, torch::kFloat64);
  return block;
}

void face_floor_uses_adjacent_density(torch::Device device) {
  auto block = make_block(device);
  auto coord = block->pcoord;
  int iu = coord->iu();
  int dipped = iu - 1;

  auto w = torch::zeros({block->phydro->peos->nvar(), coord->options->nc3(),
                         coord->options->nc2(), coord->options->nc1()},
                        torch::dtype(torch::kFloat64).device(device));
  auto rho = torch::exp(-coord->x1v / 2.);
  rho.select(-1, dipped).mul_(1.e-4);
  rho.select(-1, dipped - 1).mul_(0.2);
  w[IDN].copy_(rho);
  w[IPR].copy_(torch::exp(-coord->x1v / 2.) * 1.e5);

  auto u = block->phydro->peos->compute("W->U", {w});
  Variables vars;
  vars["hydro_w"] = torch::empty_like(w);
  block->phydro->forward(1.e-4, u, vars);

  auto mass = block->phydro->flux1()[IDN];
  auto mom = block->phydro->flux1()[IVX];
  ASSERT_EQ(mass.device(), device);
  double dipped_mass = mass.select(-1, dipped).item<double>();
  double dipped_mom = mom.select(-1, dipped).item<double>();
  double lower_mass = mass.select(-1, 4).item<double>();

  // The column is not in balance, so a face away from the dip still carries
  // an O(1) mass flux. A run that dropped every flux would not pass.
  EXPECT_NEAR(lower_mass, 4.07888, 1.e-3);
  EXPECT_LT(std::abs(dipped_mass), 1.e-9);
  EXPECT_GT(dipped_mom, 0.03042);
}

}  // namespace

TEST(hydro, face_floor_uses_adjacent_density) {
  face_floor_uses_adjacent_density(torch::kCPU);
}

TEST(hydro, face_floor_uses_adjacent_density_cuda) {
  if (!torch::cuda::is_available()) GTEST_SKIP() << "CUDA is not available";
  face_floor_uses_adjacent_density(torch::Device(torch::kCUDA, 0));
}

namespace {
struct FrozenPeek : HydroImpl {
  static auto ref() { return &FrozenPeek::_hydro_ref_x1; }
};
}

TEST(hydro, frozen_reference_is_initial_target_and_restart_persistent) {
  auto block = make_block(torch::kCPU);
  auto h = block->phydro;
  auto pc = block->pcoord;
  auto w = torch::zeros({h->peos->nvar(), pc->options->nc3(),
                        pc->options->nc2(), pc->options->nc1()}, torch::kFloat64);
  w[IDN].copy_(torch::exp(-pc->x1v / 2000.));
  w[IPR].copy_(w[IDN] * 20000.);
  auto [p, pc0, d, dc] = ((*h).*FrozenPeek::ref())(w);
  auto expected = torch::stack({p, pc0, d, dc});
  h->options->wb_density_ref() = "frozen";
  EXPECT_THROW(((*h).*FrozenPeek::ref())(w), c10::Error);
  Variables vars{{"hydro_w", w}};
  h->initialize_wb_reference(vars);
  ASSERT_TRUE(torch::equal(vars.at("wb_frozen_x1"), expected));
  w[IDN].mul_(1.1);
  w[IPR].mul_(1.2);
  auto [p1, pc1, d1, dc1] = ((*h).*FrozenPeek::ref())(w);
  EXPECT_TRUE(torch::equal(torch::stack({p1, pc1, d1, dc1}), expected));
  auto u = h->peos->compute("W->U", {w});
  auto after = h->forward(0.001, u, vars);
  EXPECT_GT(after[IVX].abs().max().item<double>(), 0.);
  EXPECT_TRUE(torch::equal(vars.at("wb_frozen_x1"), expected));

  auto restored_block = make_block(torch::kCPU);
  auto restored = restored_block->phydro;
  restored->options->wb_density_ref() = "frozen";
  Variables missing{{"hydro_w", w}};
  EXPECT_THROW(restored->initialize_wb_reference(missing, true), c10::Error);
  Variables saved{{"hydro_w", w}, {"wb_frozen_x1", expected.clone()}};
  restored->initialize_wb_reference(saved, true);
  auto [p2, pc2, d2, dc2] = ((*restored).*FrozenPeek::ref())(w);
  EXPECT_TRUE(torch::equal(torch::stack({p2, pc2, d2, dc2}), expected));
}
