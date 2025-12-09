// external
#include <gtest/gtest.h>

// snap
#include <snap/coord/coordinate.hpp>
#include <snap/coord/gnomonic_equiangle.hpp>
#include <snap/layout/cubed_sphere_layout.hpp>
#include <snap/mesh/meshblock.hpp>

// tests
#include "device_testing.hpp"

using namespace snap;

TEST(GnomonicEquiangle, area_vol) {
  auto op = MeshBlockOptionsImpl::from_yaml("test_coordinate.yaml");
  auto block = MeshBlock(op);

  auto pcoord = block->phydro->pcoord;
  auto area1 = pcoord->face_area1();
  std::cout << "area1 = \n" << area1 << std::endl;

  auto area2 = pcoord->face_area2();
  std::cout << "area2 = \n" << area2 << std::endl;

  auto area3 = pcoord->face_area3();
  std::cout << "area3 = \n" << area3 << std::endl;

  auto vol = pcoord->cell_volume();
  std::cout << "volume = \n" << vol << std::endl;
}

TEST(GnomonicEquiangle, l2g) {
  CSVel l2g[6][3];
  populate_cs_l2g_vel(l2g);

  for (int f = 0; f < 6; ++f)
    for (int i = 0; i < 3; ++i) {
      EXPECT_EQ(l2g[f][i].idx, CS_L2G_VEL[f][i].idx);
      EXPECT_EQ(l2g[f][i].sgn, CS_L2G_VEL[f][i].sgn);
    }
}

TEST_P(DeviceTest, vec_lower_raise) {
  auto op = MeshBlockOptionsImpl::from_yaml("test_coordinate.yaml");
  auto block = MeshBlock(op);
  block->to(device, dtype);

  auto pcoord = block->phydro->pcoord;

  int nc1 = pcoord->options->nc1();
  int nc2 = pcoord->options->nc2();
  int nc3 = pcoord->options->nc3();
  int nghost = pcoord->options->nghost();

  auto vel = torch::ones({3, nc3, nc2, nc1},
                         torch::TensorOptions().dtype(dtype).device(device));

  pcoord->vec_lower_(vel);
  pcoord->vec_raise_(vel);

  EXPECT_TRUE(torch::allclose(vel, torch::ones_like(vel)));
}

TEST_P(DeviceTest, contra_cart) {
  auto op = MeshBlockOptionsImpl::from_yaml("test_coordinate.yaml");
  auto block = MeshBlock(op);
  block->to(device, dtype);

  auto pcoord = block->phydro->pcoord;

  int nc1 = pcoord->options->nc1();
  int nc2 = pcoord->options->nc2();
  int nc3 = pcoord->options->nc3();
  int nghost = pcoord->options->nghost();

  auto vel_cart = torch::ones(
      {3, nc3, nc2, nc1}, torch::TensorOptions().dtype(dtype).device(device));

  auto vel = vel_cart.clone();

  pcoord->cart_to_contra_(vel);
  std::cout << "vel contravariant = \n" << vel << std::endl;
  pcoord->contra_to_cart_(vel);

  EXPECT_TRUE(torch::allclose(vel, vel_cart));
}

TEST_P(DeviceTest, usrc) {
  auto op = MeshBlockOptionsImpl::from_yaml("test_coordinate.yaml");
  auto block = MeshBlock(op);
  block->to(device, dtype);

  auto pcoord =
      std::dynamic_pointer_cast<GnomonicEquiangleImpl>(block->phydro->pcoord);
  std::cout << "usrc_LR = \n" << pcoord->usrc_LR << std::endl;
  std::cout << "usrc_BT = \n" << pcoord->usrc_BT << std::endl;
}

TEST_P(DeviceTest, interpolate_LR) {
  auto op = MeshBlockOptionsImpl::from_yaml("test_coordinate.yaml");
  auto block = MeshBlock(op);
  block->to(device, dtype);

  auto pcoord = block->phydro->pcoord;

  int nc1 = pcoord->options->nc1();
  int nc2 = pcoord->options->nc2();
  int nc3 = pcoord->options->nc3();
  int nghost = pcoord->options->nghost();

  auto var = torch::zeros({nc3, nc2, nc1},
                          torch::TensorOptions().dtype(dtype).device(device));

  // interior
  auto sub = block->part({0, 0, 0}, PartOptions().exterior(true).ndim(3));
  var.index(sub).fill_(1.0);

  // left
  sub = block->part({-1, 0, 0}, PartOptions().exterior(true).ndim(3));
  auto buf = torch::ones_like(var.index(sub)) * 2.;

  // set linear values
  for (int k = 0; k < buf.size(0); ++k)
    for (int j = 0; j < buf.size(1); ++j)
      for (int i = 0; i < buf.size(2); ++i) buf.index({k, j, i}) = j;

  std::cout << "var before = \n"
            << var.squeeze().transpose(0, 1).flip(0) << std::endl;

  var.index_put_(sub, buf);
  pcoord->fill_ghost(var, {-1, 0, 0});

  std::cout << "var after = \n"
            << var.squeeze().transpose(0, 1).flip(0) << std::endl;

  // right
  sub = block->part({1, 0, 0}, PartOptions().exterior(true).ndim(3));
  buf = torch::ones_like(var.index(sub)) * 3.;

  // set linear values
  for (int k = 0; k < buf.size(0); ++k)
    for (int j = 0; j < buf.size(1); ++j)
      for (int i = 0; i < buf.size(2); ++i) buf.index({k, j, i}) = j;

  var.index_put_(sub, buf);
  pcoord->fill_ghost(var, {1, 0, 0});

  std::cout << "var after = \n"
            << var.squeeze().transpose(0, 1).flip(0) << std::endl;

  // bottom
  sub = block->part({0, -1, 0}, PartOptions().exterior(true).ndim(3));
  buf = torch::ones_like(var.index(sub)) * 4.;

  // set linear values
  for (int k = 0; k < buf.size(0); ++k)
    for (int j = 0; j < buf.size(1); ++j)
      for (int i = 0; i < buf.size(2); ++i) buf.index({k, j, i}) = k;

  var.index_put_(sub, buf);
  pcoord->fill_ghost(var, {0, -1, 0});

  std::cout << "var after = \n"
            << var.squeeze().transpose(0, 1).flip(0) << std::endl;

  // top
  sub = block->part({0, 1, 0}, PartOptions().exterior(true).ndim(3));
  buf = torch::ones_like(var.index(sub)) * 5.;

  // set linear values
  for (int k = 0; k < buf.size(0); ++k)
    for (int j = 0; j < buf.size(1); ++j)
      for (int i = 0; i < buf.size(2); ++i) buf.index({k, j, i}) = k;

  var.index_put_(sub, buf);
  pcoord->fill_ghost(var, {0, 1, 0});

  std::cout << "var after = \n"
            << var.squeeze().transpose(0, 1).flip(0) << std::endl;
}

int main(int argc, char **argv) {
  testing::InitGoogleTest(&argc, argv);

  int result = RUN_ALL_TESTS();

  return result;
}
