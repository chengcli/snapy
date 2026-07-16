// external
#include <gtest/gtest.h>

// C/C++
#include <cmath>
#include <cstdio>
#include <fstream>
#include <string>

// POSIX
#include <unistd.h>

// snap
#include <snap/coord/coord_utils.hpp>
#include <snap/coord/coordinate.hpp>
#include <snap/coord/cubed_sphere_utils.hpp>
#include <snap/coord/gnomonic_equiangle.hpp>
#include <snap/coord/spherical_polar.hpp>
#include <snap/coord/spherical_utils.hpp>
#include <snap/layout/cubed_sphere_layout.hpp>
#include <snap/mesh/meshblock.hpp>

// tests
#include "device_testing.hpp"

using namespace snap;

namespace {

const char* gnomonic_radial_config = R"(
reference-state:
  Tref: 300.
  Pref: 1.e5

species:
  - name: dry
    composition: {O: 0.42, N: 1.56, Ar: 0.01}
    cv_R: 2.5

dynamics:
  equation-of-state:
    type: ideal-gas
    gammad: 1.4

geometry:
  type: gnomonic-equiangle
  cells: {nx1: 6, nx2: 6, nx3: 6, nghost: 2}
  bounds:
    x1min: 1.
    x1max: 2.
    x2min_pi: -0.25
    x2max_pi: 0.25
    x3min_pi: -0.25
    x3max_pi: 0.25

boundary-condition:
  external:
    x1-inner: reflecting
    x1-outer: reflecting
    x2-inner: custom
    x2-outer: custom
    x3-inner: custom
    x3-outer: custom
)";

const char* spherical_polar_config = R"(
reference-state:
  Tref: 300.
  Pref: 1.e5

species:
  - name: dry
    composition: {O: 0.42, N: 1.56, Ar: 0.01}
    cv_R: 2.5

dynamics:
  equation-of-state:
    type: ideal-gas
    gammad: 1.4

distribute:
  layout: slab
  nb2: 1
  nb3: 1
  verbose: false

geometry:
  type: spherical-polar
  cells: {nx1: 4, nx2: 4, nx3: 4, nghost: 1}
  bounds:
    x1min: 1.
    x1max: 2.
    x2min_pi: 0.25
    x2max_pi: 0.75
    x3min_pi: -0.5
    x3max_pi: 0.5

boundary-condition:
  external:
    x1-inner: reflecting
    x1-outer: reflecting
    x2-inner: reflecting
    x2-outer: reflecting
    x3-inner: periodic
    x3-outer: periodic
)";

std::string write_temp_config(char const* config) {
  char fname[] = "/tmp/test-coordinate-XXXXXX";
  int fd = mkstemp(fname);
  EXPECT_NE(fd, -1);
  if (fd != -1) close(fd);

  std::ofstream outfile(fname);
  outfile << config;
  outfile.close();
  return fname;
}

}  // namespace

TEST(GnomonicEquiangle, area_vol) {
  auto op = MeshBlockOptionsImpl::from_yaml("test_coordinate.yaml");
  auto block = MeshBlock(op);

  auto pcoord = block->pcoord;
  auto area1 = pcoord->face_area1();
  std::cout << "area1 = \n" << area1 << std::endl;

  auto area2 = pcoord->face_area2();
  std::cout << "area2 = \n" << area2 << std::endl;

  auto area3 = pcoord->face_area3();
  std::cout << "area3 = \n" << area3 << std::endl;

  auto vol = pcoord->cell_volume();
  std::cout << "volume = \n" << vol << std::endl;
}

TEST(SphericalPolar, geometry_matches_athena_reference_formulas) {
  auto fname = write_temp_config(spherical_polar_config);
  auto op = MeshBlockOptionsImpl::from_yaml(fname);
  auto block = MeshBlock(op);
  std::remove(fname.c_str());

  auto pcoord = std::dynamic_pointer_cast<SphericalPolarImpl>(block->pcoord);
  ASSERT_TRUE(pcoord != nullptr);

  auto area1 = pcoord->face_area1();
  auto area2 = pcoord->face_area2();
  auto area3 = pcoord->face_area3();
  auto vol = pcoord->cell_volume();

  int nc1 = pcoord->options->nc1();
  int nc2 = pcoord->options->nc2();

  auto x1m = pcoord->x1f.slice(0, 0, nc1);
  auto x1p = pcoord->x1f.slice(0, 1, nc1 + 1);
  auto x2m = pcoord->x2f.slice(0, 0, nc2);
  auto x2p = pcoord->x2f.slice(0, 1, nc2 + 1);

  auto expected_x1v =
      0.75 * (x1p.pow(4) - x1m.pow(4)) / (x1p.pow(3) - x1m.pow(3));
  auto expected_x2v =
      ((x2p.sin() - x2p * x2p.cos()) - (x2m.sin() - x2m * x2m.cos())) /
      (x2m.cos() - x2p.cos());
  EXPECT_TRUE(torch::allclose(pcoord->x1v, expected_x1v, 1.e-12, 1.e-12));
  EXPECT_TRUE(torch::allclose(pcoord->x2v, expected_x2v, 1.e-12, 1.e-12));

  auto radial_area23 = 0.5 * (x1p.square() - x1m.square());
  auto radial_volume = (x1p.pow(3) - x1m.pow(3)) / 3.0;
  auto polar_area1 = torch::abs(x2m.cos() - x2p.cos());
  auto sin_face = torch::abs(pcoord->x2f.sin());

  auto expected_area1 = pcoord->x1f.square().unsqueeze(0).unsqueeze(1) *
                        polar_area1.unsqueeze(0).unsqueeze(2) *
                        pcoord->dx3f.unsqueeze(1).unsqueeze(2);
  auto expected_area2 = radial_area23.unsqueeze(0).unsqueeze(1) *
                        sin_face.unsqueeze(0).unsqueeze(2) *
                        pcoord->dx3f.unsqueeze(1).unsqueeze(2);
  auto expected_area3 = (radial_area23.unsqueeze(0).unsqueeze(1) *
                         pcoord->dx2f.unsqueeze(0).unsqueeze(2))
                            .expand({pcoord->x3f.size(0), -1, -1});
  auto expected_vol = radial_volume.unsqueeze(0).unsqueeze(1) *
                      polar_area1.unsqueeze(0).unsqueeze(2) *
                      pcoord->dx3f.unsqueeze(1).unsqueeze(2);

  EXPECT_TRUE(torch::allclose(area1, expected_area1, 1.e-12, 1.e-12));
  EXPECT_TRUE(torch::allclose(area2, expected_area2, 1.e-12, 1.e-12));
  EXPECT_TRUE(torch::allclose(area3, expected_area3, 1.e-12, 1.e-12));
  EXPECT_TRUE(torch::allclose(vol, expected_vol, 1.e-12, 1.e-12));

  auto sin_m = torch::abs(x2m.sin());
  auto sin_p = torch::abs(x2p.sin());
  auto expected_src1_i =
      (radial_area23 / radial_volume).unsqueeze(0).unsqueeze(0);
  auto expected_src2_i =
      (pcoord->dx1f / ((x1m + x1p) * radial_volume)).unsqueeze(0).unsqueeze(0);
  auto expected_src1_j =
      ((sin_p - sin_m) / polar_area1).unsqueeze(0).unsqueeze(-1);
  auto expected_src2_j = ((sin_p - sin_m) / ((sin_m + sin_p) * polar_area1))
                             .unsqueeze(0)
                             .unsqueeze(-1);

  EXPECT_TRUE(
      torch::allclose(pcoord->coord_src1_i, expected_src1_i, 1.e-12, 1.e-12));
  EXPECT_TRUE(
      torch::allclose(pcoord->coord_src2_i, expected_src2_i, 1.e-12, 1.e-12));
  EXPECT_TRUE(
      torch::allclose(pcoord->coord_src1_j, expected_src1_j, 1.e-12, 1.e-12));
  EXPECT_TRUE(
      torch::allclose(pcoord->coord_src2_j, expected_src2_j, 1.e-12, 1.e-12));
  EXPECT_TRUE(
      torch::allclose(pcoord->coord_src3_j, expected_src1_j, 1.e-12, 1.e-12));
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

  auto pcoord = block->pcoord;

  int nc1 = pcoord->options->nc1();
  int nc2 = pcoord->options->nc2();
  int nc3 = pcoord->options->nc3();
  int nghost = pcoord->options->nghost();

  auto vel = torch::ones({3, nc3, nc2, nc1},
                         torch::TensorOptions().dtype(dtype).device(device));

  coord_vec_lower_(vel, pcoord->cosine_cell_kj);
  coord_vec_raise_(vel, pcoord->cosine_cell_kj);

  EXPECT_TRUE(torch::allclose(vel, torch::ones_like(vel)));
}

TEST_P(DeviceTest, contra_cart) {
  auto op = MeshBlockOptionsImpl::from_yaml("test_coordinate.yaml");
  auto block = MeshBlock(op);
  block->to(device, dtype);

  auto pcoord = block->pcoord;

  int nc1 = pcoord->options->nc1();
  int nc2 = pcoord->options->nc2();
  int nc3 = pcoord->options->nc3();
  int nghost = pcoord->options->nghost();

  auto vel_cart = torch::ones(
      {3, nc3, nc2, nc1}, torch::TensorOptions().dtype(dtype).device(device));

  auto vel = vel_cart.clone();
  auto mesh = torch::meshgrid({pcoord->x3v, pcoord->x2v, pcoord->x1v}, "ij");

  cs_cart_to_contra_(vel, mesh[0], mesh[1], 0);
  std::cout << "vel contravariant = \n" << vel << std::endl;
  cs_contra_to_cart_(vel, mesh[0], mesh[1], 0);

  EXPECT_TRUE(torch::allclose(vel, vel_cart));
}

TEST_P(DeviceTest, contra_sph) {
  auto op = MeshBlockOptionsImpl::from_yaml("test_coordinate.yaml");
  auto block = MeshBlock(op);
  block->to(device, dtype);

  auto pcoord = block->pcoord;
  int nc1 = pcoord->options->nc1();
  int nc2 = pcoord->options->nc2();
  int nc3 = pcoord->options->nc3();

  auto opts = torch::TensorOptions().dtype(dtype).device(device);
  auto mesh = torch::meshgrid({pcoord->x3v, pcoord->x2v, pcoord->x1v}, "ij");
  auto alpha = mesh[1];
  auto beta = mesh[0];

  auto vel_contra = torch::randn({3, nc3, nc2, nc1}, opts) +
                    torch::full({3, nc3, nc2, nc1}, 0.5, opts);

  for (int face = 0; face < 6; ++face) {
    auto vel = vel_contra.clone();
    cs_contra_to_sph_(vel, alpha, beta, face);
    cs_sph_to_contra_(vel, alpha, beta, face);
    EXPECT_TRUE(torch::allclose(vel, vel_contra, 1.e-4, 1.e-5))
        << "face " << face;
  }
}

TEST_P(DeviceTest, contra_sph_matches_cartesian_composition) {
  auto op = MeshBlockOptionsImpl::from_yaml("test_coordinate.yaml");
  auto block = MeshBlock(op);
  block->to(device, dtype);

  auto pcoord = block->pcoord;
  int nc1 = pcoord->options->nc1();
  int nc2 = pcoord->options->nc2();
  int nc3 = pcoord->options->nc3();

  auto opts = torch::TensorOptions().dtype(dtype).device(device);
  auto mesh = torch::meshgrid({pcoord->x3v, pcoord->x2v, pcoord->x1v}, "ij");
  auto alpha = mesh[1];
  auto beta = mesh[0];

  auto vel_contra = torch::randn({3, nc3, nc2, nc1}, opts) +
                    torch::full({3, nc3, nc2, nc1}, 0.5, opts);

  for (int face = 0; face < 6; ++face) {
    auto expected = vel_contra.clone();
    cs_contra_to_cart_(expected, alpha, beta, face);
    auto lonlat = cs_ab_to_lonlat(CS_FACE_NAMES[face], alpha, beta);
    auto theta = 0.5 * M_PI - lonlat.second;
    sph_cart_to_contra_(expected, theta, lonlat.first);

    auto actual = vel_contra.clone();
    cs_contra_to_sph_(actual, alpha, beta, face);

    EXPECT_TRUE(torch::allclose(actual, expected, 1.e-4, 1.e-5))
        << "face " << face;
  }
}

TEST_P(DeviceTest, uniform_radial_spherical_velocity_round_trips) {
  auto op = MeshBlockOptionsImpl::from_yaml("test_coordinate.yaml");
  auto block = MeshBlock(op);
  block->to(device, dtype);

  auto pcoord = block->pcoord;
  int nc1 = pcoord->options->nc1();
  int nc2 = pcoord->options->nc2();
  int nc3 = pcoord->options->nc3();

  auto opts = torch::TensorOptions().dtype(dtype).device(device);
  auto mesh = torch::meshgrid({pcoord->x3v, pcoord->x2v, pcoord->x1v}, "ij");
  auto alpha = mesh[1];
  auto beta = mesh[0];

  auto vel_sph = torch::zeros({3, nc3, nc2, nc1}, opts);
  vel_sph[VEL1].fill_(1.2345);

  for (int face = 0; face < 6; ++face) {
    auto vel = vel_sph.clone();
    cs_sph_to_contra_(vel, alpha, beta, face);
    cs_contra_to_sph_(vel, alpha, beta, face);
    EXPECT_TRUE(torch::allclose(vel, vel_sph, 1.e-4, 1.e-5)) << "face " << face;
  }
}

TEST_P(DeviceTest, usrc) {
  auto op = MeshBlockOptionsImpl::from_yaml("test_coordinate.yaml");
  auto block = MeshBlock(op);
  block->to(device, dtype);

  auto pcoord = std::dynamic_pointer_cast<GnomonicEquiangleImpl>(block->pcoord);
  std::cout << "usrc_LR = \n" << pcoord->usrc_LR << std::endl;
  std::cout << "usrc_BT = \n" << pcoord->usrc_BT << std::endl;
}

TEST_P(DeviceTest, interpolate_LR) {
  auto op = MeshBlockOptionsImpl::from_yaml("test_coordinate.yaml");
  auto block = MeshBlock(op);
  block->to(device, dtype);

  auto pcoord = block->pcoord;

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
  pcoord->interp_ghost(var, {-1, 0, 0});

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
  pcoord->interp_ghost(var, {1, 0, 0});

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
  pcoord->interp_ghost(var, {0, -1, 0});

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
  pcoord->interp_ghost(var, {0, 1, 0});

  std::cout << "var after = \n"
            << var.squeeze().transpose(0, 1).flip(0) << std::endl;
}

TEST_P(DeviceTest, flux_projection1) {
  auto op = MeshBlockOptionsImpl::from_yaml("test_coordinate.yaml");
  auto block = MeshBlock(op);
  block->to(device, dtype);

  auto pcoord = block->pcoord;

  int nc1 = pcoord->options->nc1();
  int nc2 = pcoord->options->nc2();
  int nc3 = pcoord->options->nc3();
  auto prim = torch::ones({5, nc3, nc2, nc1},
                          torch::TensorOptions().dtype(dtype).device(device));
  auto prim_ori = prim.clone();
  pcoord->prim2local1_(prim);
  pcoord->flux2global1_(prim);

  coord_vec_raise_(prim.narrow(0, IVX, 3), pcoord->cosine_cell_kj);
  EXPECT_TRUE(torch::allclose(prim, prim_ori));
}

TEST_P(DeviceTest, flux_projection2) {
  auto op = MeshBlockOptionsImpl::from_yaml("test_coordinate.yaml");
  auto block = MeshBlock(op);
  block->to(device, dtype);

  auto pcoord = block->pcoord;

  int nc1 = pcoord->options->nc1();
  int nc2 = pcoord->options->nc2();
  int nc3 = pcoord->options->nc3();
  auto prim = torch::ones({5, nc3, nc2, nc1},
                          torch::TensorOptions().dtype(dtype).device(device));
  auto prim_ori = prim.clone();
  pcoord->prim2local2_(prim);
  pcoord->flux2global2_(prim);

  auto xf = pcoord->x2f.tan().unsqueeze(0).unsqueeze(-1);
  auto y = pcoord->x3v.tan().unsqueeze(-1).unsqueeze(-1);
  auto Cf = torch::sqrt(1. + xf * xf);
  auto D = torch::sqrt(1. + y * y);
  auto cthf = -xf * y / Cf / D;

  coord_vec_raise_(prim.narrow(0, IVX, 3), cthf.narrow(1, 0, nc2));
  EXPECT_TRUE(torch::allclose(prim, prim_ori));
}

TEST_P(DeviceTest, flux_projection3) {
  auto op = MeshBlockOptionsImpl::from_yaml("test_coordinate.yaml");
  auto block = MeshBlock(op);
  block->to(device, dtype);

  auto pcoord = block->pcoord;

  int nc1 = pcoord->options->nc1();
  int nc2 = pcoord->options->nc2();
  int nc3 = pcoord->options->nc3();
  auto prim = torch::ones({5, nc3, nc2, nc1},
                          torch::TensorOptions().dtype(dtype).device(device));
  auto prim_ori = prim.clone();
  pcoord->prim2local3_(prim);
  pcoord->flux2global3_(prim);

  auto x = pcoord->x2v.tan().unsqueeze(0).unsqueeze(-1);
  auto yf = pcoord->x3f.tan().unsqueeze(-1).unsqueeze(-1);
  auto C = torch::sqrt(1. + x * x);
  auto Df = torch::sqrt(1. + yf * yf);
  auto cthf = -x * yf / C / Df;

  coord_vec_raise_(prim.narrow(0, IVX, 3), cthf.narrow(0, 0, nc3));
  EXPECT_TRUE(torch::allclose(prim, prim_ori));
}

TEST_P(DeviceTest, radial_source_uses_face_pressure_in_x1_momentum) {
  auto fname = write_temp_config(gnomonic_radial_config);
  auto op = MeshBlockOptionsImpl::from_yaml(fname);
  auto block = MeshBlock(op);
  block->to(device, dtype);
  std::remove(fname.c_str());

  auto pcoord = block->pcoord;
  int nc1 = pcoord->options->nc1();
  int nc2 = pcoord->options->nc2();
  int nc3 = pcoord->options->nc3();
  auto opts = torch::TensorOptions().dtype(dtype).device(device);

  auto prim_lo = torch::zeros({5, nc3, nc2, nc1}, opts);
  auto prim_hi = torch::zeros_like(prim_lo);
  prim_lo[IDN].fill_(1.0);
  prim_hi[IDN].fill_(1.0);
  prim_lo[IPR].fill_(3.0);
  prim_hi[IPR].fill_(11.0);

  auto flux1 = torch::zeros_like(prim_lo);
  auto face_pressure = torch::full({nc3, nc2, nc1}, 5.0, opts);

  auto div_lo = pcoord->forward(prim_lo, flux1, torch::Tensor(),
                                torch::Tensor(), face_pressure);
  auto div_hi = pcoord->forward(prim_hi, flux1, torch::Tensor(),
                                torch::Tensor(), face_pressure);

  EXPECT_TRUE(torch::allclose(div_lo[IVX], div_hi[IVX], 1.e-8, 1.e-8));
}

TEST_P(DeviceTest, radial_source_preserves_face_pressure_gradient) {
  auto fname = write_temp_config(gnomonic_radial_config);
  auto op = MeshBlockOptionsImpl::from_yaml(fname);
  auto block = MeshBlock(op);
  block->to(device, dtype);
  std::remove(fname.c_str());

  auto pcoord = block->pcoord;
  int nc1 = pcoord->options->nc1();
  int nc2 = pcoord->options->nc2();
  int nc3 = pcoord->options->nc3();
  int si = pcoord->il();
  int ei = pcoord->iu() + 1;
  auto opts = torch::TensorOptions().dtype(dtype).device(device);

  auto prim = torch::zeros({5, nc3, nc2, nc1}, opts);
  prim[IDN].fill_(1.0);
  auto face_pressure = pcoord->x1f.slice(0, 0, nc1)
                           .to(opts)
                           .view({1, 1, nc1})
                           .expand({nc3, nc2, nc1});
  auto flux1 = torch::zeros_like(prim);
  flux1[IVX].copy_(face_pressure);

  auto div = pcoord->forward(prim, flux1, torch::Tensor(), torch::Tensor(),
                             face_pressure);
  auto radial_div = div[IVX].slice(-1, si, ei);
  auto expected = (face_pressure.slice(-1, si + 1, ei + 1) -
                   face_pressure.slice(-1, si, ei)) /
                  pcoord->dx1f.slice(0, si, ei);

  EXPECT_TRUE(torch::allclose(radial_div, expected, 1.e-6, 1.e-6))
      << "radial_div=" << radial_div << "\nexpected=" << expected;
}

TEST_P(DeviceTest,
       spherical_polar_radial_source_preserves_face_pressure_gradient) {
  auto fname = write_temp_config(spherical_polar_config);
  auto op = MeshBlockOptionsImpl::from_yaml(fname);
  auto block = MeshBlock(op);
  block->to(device, dtype);
  std::remove(fname.c_str());

  auto pcoord = block->pcoord;
  int nc1 = pcoord->options->nc1();
  int nc2 = pcoord->options->nc2();
  int nc3 = pcoord->options->nc3();
  int si = pcoord->il();
  int ei = pcoord->iu() + 1;
  auto opts = torch::TensorOptions().dtype(dtype).device(device);

  auto prim = torch::zeros({5, nc3, nc2, nc1}, opts);
  prim[IDN].fill_(1.0);
  auto face_pressure = pcoord->x1f.slice(0, 0, nc1)
                           .to(opts)
                           .view({1, 1, nc1})
                           .expand({nc3, nc2, nc1});
  auto flux1 = torch::zeros_like(prim);
  flux1[IVX].copy_(face_pressure);
  auto flux2 = torch::zeros_like(prim);
  auto flux3 = torch::zeros_like(prim);

  auto div = pcoord->forward(prim, flux1, flux2, flux3, face_pressure);
  auto radial_div = div[IVX].slice(-1, si, ei);
  auto expected = (face_pressure.slice(-1, si + 1, ei + 1) -
                   face_pressure.slice(-1, si, ei)) /
                  pcoord->dx1f.slice(0, si, ei);

  EXPECT_TRUE(torch::allclose(radial_div, expected, 1.e-6, 1.e-6))
      << "radial_div=" << radial_div << "\nexpected=" << expected;
}

TEST_P(DeviceTest,
       spherical_polar_radial_source_uses_face_pressure_in_x1_momentum) {
  auto fname = write_temp_config(spherical_polar_config);
  auto op = MeshBlockOptionsImpl::from_yaml(fname);
  auto block = MeshBlock(op);
  block->to(device, dtype);
  std::remove(fname.c_str());

  auto pcoord = block->pcoord;
  int nc1 = pcoord->options->nc1();
  int nc2 = pcoord->options->nc2();
  int nc3 = pcoord->options->nc3();
  auto opts = torch::TensorOptions().dtype(dtype).device(device);

  auto prim_lo = torch::zeros({5, nc3, nc2, nc1}, opts);
  auto prim_hi = torch::zeros_like(prim_lo);
  prim_lo[IDN].fill_(1.0);
  prim_hi[IDN].fill_(1.0);
  prim_lo[IPR].fill_(3.0);
  prim_hi[IPR].fill_(11.0);

  auto flux1 = torch::zeros_like(prim_lo);
  auto flux2 = torch::zeros_like(prim_lo);
  auto flux3 = torch::zeros_like(prim_lo);
  auto face_pressure = torch::full({nc3, nc2, nc1}, 5.0, opts);

  auto div_no_face_lo = pcoord->forward(prim_lo, flux1, flux2, flux3);
  auto div_no_face_hi = pcoord->forward(prim_hi, flux1, flux2, flux3);
  auto div_face_lo =
      pcoord->forward(prim_lo, flux1, flux2, flux3, face_pressure);
  auto div_face_hi =
      pcoord->forward(prim_hi, flux1, flux2, flux3, face_pressure);

  EXPECT_FALSE(
      torch::allclose(div_no_face_lo[IVX], div_no_face_hi[IVX], 1.e-6, 1.e-6));
  EXPECT_TRUE(
      torch::allclose(div_face_lo[IVX], div_face_hi[IVX], 1.e-6, 1.e-6));
}

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);

  int result = RUN_ALL_TESTS();

  return result;
}
