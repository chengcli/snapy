// external
#include <gtest/gtest.h>

// C/C++
#include <vector>

// snap
#include <snap/implicit/forward_backward_impl.h>

namespace {

double column_integral(std::vector<double> const& u,
                       std::vector<double> const& vol, int var, int stride1,
                       int il, int iu) {
  double sum = 0;
  for (int i = il; i <= iu; ++i) sum += vol[i] * u[var * stride1 + i];
  return sum;
}

}  // namespace

TEST(backward_substitution, conserves_constituent_column_tendencies) {
  constexpr int nlayer = 4;
  constexpr int ny = 2;
  constexpr int nhydro = snap::ICY + ny;
  constexpr int stride1 = nlayer;
  constexpr int stride2 = 1;
  constexpr int il = 0;
  constexpr int iu = nlayer - 1;

  std::vector<double> du(nhydro * nlayer, 0.);
  std::vector<double> w(nhydro * nlayer, 0.);
  std::vector<double> vol = {1.0, 2.0, 1.5, 0.5};
  std::vector<double> corr = {0.4, -0.2, 0.8, -0.1};

  std::vector<Eigen::Matrix<double, 3, 3>> a(nlayer);
  std::vector<Eigen::Matrix<double, 3, 1>> delta(nlayer);

  for (int i = il; i <= iu; ++i) {
    double dry = 1.0 + 0.2 * i;
    double vapor = 0.10 + 0.03 * i;
    double cloud = 0.05 + 0.01 * i;

    du[snap::IDN * stride1 + i] = dry;
    du[snap::IVX * stride1 + i] = 10.0 + i;
    du[snap::IPR * stride1 + i] = 20.0 + i;
    du[snap::ICY * stride1 + i] = vapor;
    du[(snap::ICY + 1) * stride1 + i] = cloud;

    w[snap::ICY * stride1 + i] = 0.10 + 0.02 * i;
    w[(snap::ICY + 1) * stride1 + i] = 0.05 + 0.01 * i;

    a[i].setZero();
    delta[i] << dry + vapor + cloud + corr[i], 100.0 + i, 200.0 + i;
  }

  auto original = du;

  snap::BackwardSubstitution<double, 3>(du.data(), w.data(), a.data(),
                                        delta.data(), vol.data(), il, iu, 0, ny,
                                        stride1, stride2, true, true);

  int vars[] = {snap::IDN, snap::ICY, snap::ICY + 1};
  for (int var : vars) {
    EXPECT_NEAR(column_integral(du, vol, var, stride1, il, iu),
                column_integral(original, vol, var, stride1, il, iu), 1.e-12);
  }

  for (int var : vars) {
    double first_offset = 0.;
    for (int i = il; i <= iu; ++i) {
      double fraction = 0.;
      if (var == snap::IDN) {
        fraction =
            1. - w[snap::ICY * stride1 + i] - w[(snap::ICY + 1) * stride1 + i];
      } else {
        fraction = w[var * stride1 + i];
      }

      double redistributed = original[var * stride1 + i] + fraction * corr[i];
      double offset = du[var * stride1 + i] - redistributed;
      if (i == il) first_offset = offset;
      EXPECT_NEAR(offset, first_offset, 1.e-12);
    }
  }
}

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
