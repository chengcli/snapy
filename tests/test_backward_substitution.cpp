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
  std::vector<double> corr = {0.4, -0.2, 0.1, -0.3};

  std::vector<Eigen::Matrix<double, 3, 3>> a(nlayer);
  std::vector<Eigen::Matrix<double, 3, 1>> delta(nlayer);

  for (int i = il; i <= iu; ++i) {
    double vapor_frac = 0.10 + 0.02 * i;
    double cloud_frac = 0.05 + 0.01 * i;
    double dry_frac = 1. - vapor_frac - cloud_frac;
    double explicit_total = 1.5 + 0.2 * i;

    du[snap::IDN * stride1 + i] = explicit_total * dry_frac;
    du[snap::IVX * stride1 + i] = 10.0 + i;
    du[snap::IPR * stride1 + i] = 20.0 + i;
    du[snap::ICY * stride1 + i] = explicit_total * vapor_frac;
    du[(snap::ICY + 1) * stride1 + i] = explicit_total * cloud_frac;

    w[snap::ICY * stride1 + i] = vapor_frac;
    w[(snap::ICY + 1) * stride1 + i] = cloud_frac;

    a[i].setZero();
    delta[i] << explicit_total + corr[i], 100.0 + i, 200.0 + i;
  }

  auto original = du;
  std::vector<double> implicit_total(nlayer, 0.);
  double denom = 0.;
  for (int i = il; i <= iu; ++i) {
    implicit_total[i] = delta[i](0);
    double weighted_implicit = implicit_total[i] * vol[i];
    denom += weighted_implicit * weighted_implicit;
  }

  snap::BackwardSubstitution<double, 3>(du.data(), w.data(), a.data(),
                                        delta.data(), vol.data(), il, iu, 0, ny,
                                        stride1, stride2, true, true);

  int vars[] = {snap::IDN, snap::ICY, snap::ICY + 1};
  for (int var : vars) {
    EXPECT_NEAR(column_integral(du, vol, var, stride1, il, iu),
                column_integral(original, vol, var, stride1, il, iu), 1.e-5);
  }

  for (int var : vars) {
    double species_struct = 0.;
    for (int i = il; i <= iu; ++i) {
      double fraction = 0.;
      if (var == snap::IDN) {
        fraction =
            1. - w[snap::ICY * stride1 + i] - w[(snap::ICY + 1) * stride1 + i];
      } else {
        fraction = w[var * stride1 + i];
      }

      species_struct +=
          (original[snap::IDN * stride1 + i] +
           original[snap::ICY * stride1 + i] +
           original[(snap::ICY + 1) * stride1 + i] - implicit_total[i]) *
          vol[i] * fraction;
    }

    for (int i = il; i <= iu; ++i) {
      double fraction = 0.;
      if (var == snap::IDN) {
        fraction =
            1. - w[snap::ICY * stride1 + i] - w[(snap::ICY + 1) * stride1 + i];
      } else {
        fraction = w[var * stride1 + i];
      }

      double weighted_implicit = implicit_total[i] * vol[i];
      double expected = implicit_total[i] *
                        (fraction + weighted_implicit * species_struct / denom);
      EXPECT_NEAR(du[var * stride1 + i], expected, 1.e-5);
    }
  }

  for (int i = il; i <= iu; ++i) {
    double constituent_sum = du[snap::IDN * stride1 + i] +
                             du[snap::ICY * stride1 + i] +
                             du[(snap::ICY + 1) * stride1 + i];
    EXPECT_NEAR(constituent_sum, implicit_total[i], 1.e-5);
  }
}

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
