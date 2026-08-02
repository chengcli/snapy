// external
#include <gtest/gtest.h>
#include <yaml-cpp/yaml.h>

// C/C++
#include <vector>

// snap
#include <snap/implicit/vic_redistribute_impl.h>

#include <snap/implicit/implicit_hydro.hpp>

namespace {

double column_integral(std::vector<double> const& u,
                       std::vector<double> const& vol, int var, int stride1,
                       int il, int iu) {
  double sum = 0;
  for (int i = il; i <= iu; ++i) sum += vol[i] * u[var * stride1 + i];
  return sum;
}

}  // namespace

TEST(vic_redistribution, conserves_constituent_column_tendencies) {
  constexpr int nlayer = 4;
  constexpr int ny = 3;
  constexpr int nhydro = snap::ICY + ny;
  constexpr int stride1 = nlayer;
  constexpr int stride2 = 1;
  constexpr int il = 0;
  constexpr int iu = nlayer - 1;

  std::vector<double> du(nhydro * nlayer, 0.);
  std::vector<double> w(nhydro * nlayer, 0.);
  std::vector<double> vol = {1.0, 2.0, 1.5, 0.5};
  std::vector<double> corr = {0.4, -0.2, 0.1, -0.3};
  std::vector<double> rho = {3.0, 2.5, 4.0, 1.8};

  std::vector<Eigen::Matrix<double, 3, 3>> a(nlayer);
  std::vector<Eigen::Matrix<double, 3, 1>> delta(nlayer);
  std::vector<double> mass_fix(nhydro * nlayer, 0.);
  for (int i = il; i <= iu; ++i) {
    double vapor_frac = 0.10 + 0.02 * i;
    double cloud_frac = 0.05 + 0.01 * i;
    double tracer_frac = 0.02 + 0.005 * i;
    double dry_frac = 1. - vapor_frac - cloud_frac - tracer_frac;
    double explicit_total = 1.5 + 0.2 * i;

    du[snap::IDN * stride1 + i] = explicit_total * dry_frac;
    du[snap::IVX * stride1 + i] = 10.0 + i;
    du[snap::IPR * stride1 + i] = 20.0 + i;
    du[snap::ICY * stride1 + i] = explicit_total * vapor_frac;
    du[(snap::ICY + 1) * stride1 + i] = explicit_total * cloud_frac;
    du[(snap::ICY + 2) * stride1 + i] = explicit_total * tracer_frac;

    w[snap::ICY * stride1 + i] = vapor_frac;
    w[(snap::ICY + 1) * stride1 + i] = cloud_frac;
    w[(snap::ICY + 2) * stride1 + i] = tracer_frac;
    w[snap::IDN * stride1 + i] = rho[i];

    a[i].setZero();
    delta[i] << explicit_total + corr[i], 100.0 + i, 200.0 + i;
  }

  auto original = du;
  snap::vic_backward_substitute<double, 3>(a.data(), delta.data(), il, iu);
  snap::vic_constituent_column<double, 3>(du.data(), w.data(), mass_fix.data(),
                                          delta.data(), vol.data(), nlayer, 0,
                                          ny, stride1, stride2);
  for (int i = il; i <= iu; ++i) {
    snap::vic_redistribute_cell<double, 3>(
        du.data(), mass_fix.data(), delta.data(), i, 0, ny, stride1, stride2);
  }

  int constituent_vars[] = {snap::IDN, snap::ICY, snap::ICY + 1, snap::ICY + 2};
  for (int var : constituent_vars) {
    EXPECT_NEAR(column_integral(du, vol, var, stride1, il, iu),
                column_integral(original, vol, var, stride1, il, iu), 1.e-12);
  }

  std::vector<double> expected_face_mass = {0., -0.4, 0., -0.15};
  for (int i = il; i <= iu; ++i) {
    EXPECT_NEAR(mass_fix[snap::IVX * stride1 + i], expected_face_mass[i],
                1.e-12);

    double constituent_total = du[snap::IDN * stride1 + i];
    for (int n = 0; n < ny; ++n) {
      constituent_total += du[(snap::ICY + n) * stride1 + i];
      double final_mass = (rho[i] * w[(snap::ICY + n) * stride1 + i] +
                           du[(snap::ICY + n) * stride1 + i]) *
                          vol[i];
      EXPECT_GE(final_mass, -1.e-12);
    }
    EXPECT_NEAR(constituent_total, delta[i](0), 1.e-12);
  }
}

TEST(vic_redistribution, dry_only_transport_is_conservative_and_clamped) {
  constexpr int nlayer = 2;
  constexpr int ny = 0;
  constexpr int nhydro = snap::ICY;
  constexpr int stride1 = nlayer;
  constexpr int stride2 = 1;

  std::vector<double> du(nhydro * nlayer, 0.);
  std::vector<double> w(nhydro * nlayer, 0.);
  std::vector<double> vol(nlayer, 1.);
  std::vector<double> mass_fix(nhydro * nlayer, 0.);
  std::vector<Eigen::Matrix<double, 3, 1>> delta(nlayer);

  w[snap::IDN * stride1] = 1.;
  w[snap::IDN * stride1 + 1] = 1.;
  du[snap::IDN * stride1] = -0.75;
  delta[0] << -2.75, 0., 0.;
  delta[1] << 2., 0., 0.;

  auto original = du;
  snap::vic_constituent_column<double, 3>(du.data(), w.data(), mass_fix.data(),
                                          delta.data(), vol.data(), nlayer, 0,
                                          ny, stride1, stride2);
  for (int i = 0; i < nlayer; ++i) {
    snap::vic_redistribute_cell<double, 3>(
        du.data(), mass_fix.data(), delta.data(), i, 0, ny, stride1, stride2);
  }

  EXPECT_NEAR(mass_fix[snap::IVX * stride1 + 1], 2., 1.e-12);
  EXPECT_NEAR(mass_fix[snap::IDN * stride1], -0.25, 1.e-12);
  EXPECT_NEAR(mass_fix[snap::IDN * stride1 + 1], 0.25, 1.e-12);
  EXPECT_NEAR(column_integral(du, vol, snap::IDN, stride1, 0, 1),
              column_integral(original, vol, snap::IDN, stride1, 0, 1), 1.e-12);
  EXPECT_NEAR(w[snap::IDN * stride1] + du[snap::IDN * stride1], 0., 1.e-12);
}

TEST(implicit_options, parses_implicit_scheme_bits) {
  auto partial = snap::ImplicitOptionsImpl::from_yaml(YAML::Load("1"));
  ASSERT_TRUE(partial);
  EXPECT_EQ(partial->scheme(), 1);
  EXPECT_EQ(partial->size(), 3);
  EXPECT_EQ(partial->type(), "vic-partial");

  auto full = snap::ImplicitOptionsImpl::from_yaml(YAML::Load("9"));
  ASSERT_TRUE(full);
  EXPECT_EQ(full->scheme(), 9);
  EXPECT_EQ(full->size(), 5);
  EXPECT_EQ(full->type(), "vic-full");
}

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
