#include <gtest/gtest.h>
#include <snap/layout/cubed_sphere_constants.h>

#include <snap/eb/geometry.hpp>

using snap::EBGeometry;

namespace {

torch::Tensor axis(std::initializer_list<double> x) {
  return torch::tensor(std::vector<double>(x), torch::kFloat64);
}

EBGeometry import(const EBGeometry& g, const torch::Tensor& x,
                  const torch::Tensor& y, const torch::Tensor& z) {
  return EBGeometry::from_metrics(g.volume_fraction, g.cell_centroid,
                                  g.face_fraction, g.face_centroid,
                                  g.wall_area_vector, g.wall_centroid, x, y, z);
}

TEST(EBGeometry, AlignedWallClosesFaceAndRetainsFullFluidVolume) {
  auto x = axis({0, 1, 2, 3}), y = axis({5, 7}), z = axis({8, 11});
  auto phi = (x - 1).expand({2, 2, 4});
  auto g = EBGeometry::from_level_set(phi, x, y, z);
  EXPECT_TRUE(torch::equal(g.volume_fraction.flatten(), axis({0, 1, 1})));
  EXPECT_TRUE(torch::equal(g.flags.flatten(), torch::tensor({2, 1, 0})));
  EXPECT_TRUE(torch::equal(g.face_fraction[0].flatten(), axis({0, 0, 1, 1})));
  EXPECT_TRUE(torch::allclose(g.wall_area_vector.select(3, 1).flatten(),
                              axis({-6, 0, 0})));
  EXPECT_TRUE(torch::allclose(g.wall_centroid.select(3, 1).flatten(),
                              axis({1, 6, 9.5})));
  EXPECT_TRUE(torch::equal(import(g, x, y, z).flags, g.flags));
}

TEST(EBGeometry, ObliqueTetrahedronMomentsIncludePhysicalStretch) {
  auto x = axis({11, 13}), y = axis({20, 23}), z = axis({-8, -4});
  auto phi = 1 - (x - 11) / 2 - (y.view({2, 1}) - 20) / 3 -
             (z.view({2, 1, 1}) + 8) / 4;
  auto g = EBGeometry::from_level_set(phi, x, y, z);
  EXPECT_NEAR(g.volume_fraction.item<double>(), 1.0 / 6, 1e-14);
  EXPECT_TRUE(
      torch::allclose(g.cell_centroid.flatten(), axis({11.5, 20.75, -7})));
  EXPECT_TRUE(torch::allclose(g.wall_centroid.flatten(),
                              axis({11 + 2. / 3, 21, -8 + 4. / 3})));
  EXPECT_TRUE(torch::allclose(g.wall_area_vector.flatten(), axis({6, 4, 3})));
  EXPECT_TRUE(torch::allclose(g.face_centroid[0].select(3, 0).flatten(),
                              axis({11, 21, -8 + 4. / 3})));
  auto complement = EBGeometry::from_level_set(-phi, x, y, z);
  EXPECT_NEAR(complement.volume_fraction.item<double>(), 5.0 / 6, 1e-14);
  EXPECT_TRUE(
      torch::allclose(g.cell_centroid * g.volume_fraction +
                          complement.cell_centroid * complement.volume_fraction,
                      axis({12, 21.5, -6}).view({3, 1, 1, 1})));
}

TEST(EBGeometry, TinyVolumesAndSharedFaceCentroids) {
  auto x = axis({0, 1, 2}), yz = axis({0, 1});
  auto phi = 1 + 1e-7 - x - yz.view({2, 1}) - yz.view({2, 1, 1});
  auto g = EBGeometry::from_level_set(phi, x, yz, yz);
  double edge = 1 + 1e-7 - 1;
  EXPECT_NEAR(g.volume_fraction[0][0][1].item<double>(), edge * edge * edge / 6,
              1e-35);
  EXPECT_NEAR(g.face_fraction[0][0][0][1].item<double>(), edge * edge / 2,
              1e-28);
  EXPECT_TRUE(torch::equal(import(g, x, yz, yz).flags, g.flags));
}

TEST(EBGeometry, RejectsInconsistentMetricsAndAmbiguousSheets) {
  auto x = axis({0, 1}), y = x, z = x;
  auto phi = (x - 0.25).expand({2, 2, 2});
  auto g = EBGeometry::from_level_set(phi, x, y, z);
  g.wall_area_vector.zero_();
  EXPECT_THROW(import(g, x, y, z), c10::Error);
  g = EBGeometry::from_level_set(phi, x, y, z);
  g.face_centroid[1].select(0, 1).fill_(0.5);
  EXPECT_THROW(import(g, x, y, z), c10::Error);
  auto ambiguous = ((2 * x - 1) * (2 * y.view({2, 1}) - 1)).expand({2, 2, 2});
  EXPECT_THROW(EBGeometry::from_level_set(ambiguous, x, y, z), c10::Error);
}

EBGeometry import_sphere(const EBGeometry& g, const torch::Tensor& r,
                         const torch::Tensor& a, const torch::Tensor& b) {
  return EBGeometry::from_cubed_sphere_metrics(
      g.volume_fraction, g.cell_centroid, g.face_fraction, g.face_centroid,
      g.face_area_vector, g.wall_area_vector, g.wall_centroid, r, a, b,
      g.panel);
}

TEST(EBGeometry, SixPanelRadialWallHasExactSphericalMetrics) {
  const double pi = std::acos(-1.);
  auto r = axis({2, 4}), a = axis({-pi / 4, 0, pi / 4}),
       phi = (r - 3).expand({3, 3, 2});
  double total = 0;
  auto sumwall = torch::zeros({3}, torch::kFloat64);
  for (int p = 0; p < 6; ++p) {
    auto g = EBGeometry::from_cubed_sphere_level_set(phi, r, a, a, p);
    EXPECT_EQ(g.panel, p);
    EXPECT_TRUE(g.flags.eq(1).all().item<bool>());
    EXPECT_TRUE(torch::allclose(g.volume_fraction,
                                torch::full_like(g.volume_fraction, 37. / 56.),
                                1e-13, 1e-13));
    EXPECT_TRUE(torch::allclose(g.face_fraction[1],
                                torch::full_like(g.face_fraction[1], 7. / 12.),
                                1e-13, 1e-13));
    EXPECT_NEAR(g.background_face_area[0].select(-1, 1).sum().item<double>(),
                16 * 4 * pi / 6, 1e-13);
    total += (g.background_volume * g.volume_fraction).sum().item<double>();
    sumwall += g.wall_area_vector.flatten(1).sum(1);
    EXPECT_TRUE(torch::equal(import_sphere(g, r, a, a).flags, g.flags));
  }
  EXPECT_NEAR(total, (64. - 27.) * 4 * pi / 3, 2e-12);
  EXPECT_LT(sumwall.abs().max().item<double>(), 1e-13);
}

TEST(EBGeometry, ReversedPanelEdgeHasOneCutFace) {
  // +Y bottom and -Z right share an edge in reverse tangential order. The
  // physical nodal level set has a nonplanar intersection in (r,tan angle).
  const double pi = std::acos(-1.);
  auto r = axis({2, 3}), a = axis({-pi / 4, -0.35, 0, 0.35, pi / 4});
  auto make = [&](int panel) {
    auto phi = torch::empty({5, 5, 2}, torch::kFloat64);
    auto p = phi.accessor<double, 3>();
    for (int k = 0; k < 5; ++k)
      for (int j = 0; j < 5; ++j)
        for (int i = 0; i < 2; ++i) {
          double u = std::tan(a[j].item<double>()),
                 v = std::tan(a[k].item<double>());
          double h = r[i].item<double>() / std::hypot(1., u, v);
          double x = h * (panel == 1 ? -u : v), z = h * (panel == 1 ? v : -1);
          p[k][j][i] = r[i].item<double>() - 2.45 + 0.2 * x * z;
        }
    return EBGeometry::from_cubed_sphere_level_set(phi, r, a, a, panel);
  };
  auto y = make(1), z = make(5);
  auto yf = y.face_fraction[2].select(0, 0),
       zf = z.face_fraction[1].select(1, 4).flip({0});
  EXPECT_TRUE(torch::allclose(yf, zf, 1.e-10, 1.e-12));
  EXPECT_TRUE(torch::allclose(y.face_centroid[2].select(1, 0),
                              z.face_centroid[1].select(2, 4).flip({1}), 1.e-10,
                              1.e-12));
  EXPECT_TRUE(torch::allclose(y.face_area_vector[2].select(1, 0),
                              z.face_area_vector[1].select(2, 4).flip({1}),
                              1.e-10, 1.e-12));
  EXPECT_TRUE(torch::equal(import_sphere(y, r, a, a).flags, y.flags));
  y.wall_area_vector.neg_();
  EXPECT_THROW(import_sphere(y, r, a, a), c10::Error);
}

TEST(EBGeometry, ObliqueShellCutImportsWithGhostMetricsOnAllPanels) {
  const double pi = std::acos(-1.);
  auto r = torch::linspace(9., 12., 10, torch::kFloat64);
  auto a = torch::arange(-3, 16, torch::kFloat64) * (pi / 24) - pi / 4;
  for (int panel = 0; panel < 6; ++panel) {
    auto phi = torch::empty({a.numel(), a.numel(), r.numel()}, torch::kFloat64);
    auto data = phi.accessor<double, 3>();
    for (int k = 0; k < a.numel(); ++k)
      for (int j = 0; j < a.numel(); ++j) {
        double x, y, z;
        snap::cs_face_xyz_from_tan(panel, std::tan(a[j].item<double>()),
                                   std::tan(a[k].item<double>()), &x, &y, &z);
        for (int n = 0; n < r.numel(); ++n)
          data[k][j][n] = (x - y + 0.12 * z) / std::hypot(x, y, z) - 0.075;
      }
    auto g = EBGeometry::from_cubed_sphere_level_set(phi, r, a, a, panel);
    EXPECT_TRUE(torch::equal(import_sphere(g, r, a, a).flags, g.flags));
  }
}

TEST(EBGeometry, TinyAngularApertureHasPhysicalCentroid) {
  auto r = axis({2., 3.}), a = axis({0., 0.2}), b = axis({0., 0.3});
  double eps = 1.e-6;
  auto phi = (eps - a.tan().view({1, 2, 1}) - b.tan().view({2, 1, 1}))
                 .expand({2, 2, 2});
  auto g = EBGeometry::from_cubed_sphere_level_set(phi, r, a, b, 0);
  double omega = eps * eps / 2;
  EXPECT_NEAR((g.volume_fraction * g.background_volume).item<double>(),
              (27. - 8.) / 3 * omega, 1.e-20);
  EXPECT_TRUE(torch::equal(import_sphere(g, r, a, b).flags, g.flags));
}

}  // namespace

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
