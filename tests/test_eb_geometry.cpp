#include <gtest/gtest.h>

#include <snap/eb/geometry.hpp>

using snap::EBGeometry;

namespace {

torch::Tensor axis(std::initializer_list<double> x) {
  return torch::tensor(std::vector<double>(x), torch::kFloat64);
}

EBGeometry import(const EBGeometry &g, const torch::Tensor &x,
                  const torch::Tensor &y, const torch::Tensor &z) {
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

}  // namespace

int main(int argc, char **argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
