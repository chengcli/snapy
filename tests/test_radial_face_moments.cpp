// external
#include <gtest/gtest.h>

// snap
#include <snap/coord/coordinate.hpp>

using namespace snap;

// #289: the curved-grid (spherical-polar and cubed-sphere) x1 moments of an
// x2/x3 face, against values derived from the exact integrals, not from the
// closed forms they test. On a cell [rm, rp] the face measure is r dr and the
// cell measure r^2 dr, so with I_k = int_rm^rp r^k dr:
//   r_c = I_2 / I_1                    (face, area-weighted centroid)
//   r_v = I_3 / I_2                    (cell, volume-weighted centroid)
//   radial_face_centroid_shift_  = r_v - r_c
//   radial_face_moment2_         = (I_3 - 2 r_c I_2 + r_c^2 I_1) / I_1
//                                  (area-weighted second central moment)
// Cells used (exact rationals):
//   [1.5, 2.5]: I = (2, 49/12, 17/2)   -> shift 47/1176, moment2 47/576
//   [3, 5]:     I = (8, 98/3, 136)     -> shift 47/588,  moment2 47/144
//   [1, 2]:     I = (3/2, 7/3, 15/4)   -> shift 13/252,  moment2 13/162
// Either function returning 0 (or its Cartesian value h^2/12, resp. 0) fails.

namespace {

struct Cell {
  double rm, rp, shift, moment2;
};

const Cell kCells[] = {
    {1.5, 2.5, 47. / 1176., 47. / 576.},
    {3.0, 5.0, 47. / 588., 47. / 144.},
    {1.0, 2.0, 13. / 252., 13. / 162.},
};

// the same quantities straight from the integrals, in double, as a cross-check
// of the hand-derived rationals above
void from_integrals(double rm, double rp, double* shift, double* moment2) {
  double i1 = (rp * rp - rm * rm) / 2.;
  double i2 = (rp * rp * rp - rm * rm * rm) / 3.;
  double i3 = (rp * rp * rp * rp - rm * rm * rm * rm) / 4.;
  double rc = i2 / i1, rv = i3 / i2;
  *shift = rv - rc;
  *moment2 = (i3 - 2. * rc * i2 + rc * rc * i1) / i1;
}

}  // namespace

TEST(RadialFaceMoments, the_rationals_are_the_integrals) {
  for (auto const& c : kCells) {
    double s, m;
    from_integrals(c.rm, c.rp, &s, &m);
    EXPECT_NEAR(s, c.shift, 1e-13 * c.shift) << c.rm << " " << c.rp;
    EXPECT_NEAR(m, c.moment2, 1e-13 * c.moment2) << c.rm << " " << c.rp;
  }
}

TEST(RadialFaceMoments, closed_forms_match_the_exact_integrals) {
  for (auto const& c : kCells) {
    auto x1f = torch::tensor({c.rm, c.rp}, torch::kFloat64);
    double shift =
        CoordinateImpl::radial_face_centroid_shift_(x1f, 1).item<double>();
    double moment2 =
        CoordinateImpl::radial_face_moment2_(x1f, 1).item<double>();
    EXPECT_NEAR(shift, c.shift, 1e-14 * c.shift)
        << "radial_face_centroid_shift_ on [" << c.rm << ", " << c.rp << "]";
    EXPECT_NEAR(moment2, c.moment2, 1e-14 * c.moment2)
        << "radial_face_moment2_ on [" << c.rm << ", " << c.rp << "]";
  }
}

TEST(RadialFaceMoments, one_value_per_cell_on_a_multi_cell_column) {
  // cells [1, 1.5], [1.5, 2.5], [2.5, 3], [3, 5]; the second and fourth are
  // the tabulated ones, so the per-cell indexing is checked too
  auto x1f = torch::tensor({1.0, 1.5, 2.5, 3.0, 5.0}, torch::kFloat64);
  auto shift = CoordinateImpl::radial_face_centroid_shift_(x1f, 4);
  auto moment2 = CoordinateImpl::radial_face_moment2_(x1f, 4);
  ASSERT_EQ(shift.numel(), 4);
  ASSERT_EQ(moment2.numel(), 4);
  EXPECT_NEAR(shift[1].item<double>(), 47. / 1176., 1e-14);
  EXPECT_NEAR(moment2[1].item<double>(), 47. / 576., 1e-14);
  EXPECT_NEAR(shift[3].item<double>(), 47. / 588., 1e-14);
  EXPECT_NEAR(moment2[3].item<double>(), 47. / 144., 1e-14);
  for (int i = 0; i < 4; ++i) {
    double s, m;
    double rm = x1f[i].item<double>(), rp = x1f[i + 1].item<double>();
    from_integrals(rm, rp, &s, &m);
    EXPECT_NEAR(shift[i].item<double>(), s, 1e-12 * s) << "cell " << i;
    EXPECT_NEAR(moment2[i].item<double>(), m, 1e-12 * m) << "cell " << i;
  }
}

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
