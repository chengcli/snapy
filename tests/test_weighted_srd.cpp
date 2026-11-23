#include <gtest/gtest.h>

#include <snap/eb/geometry.hpp>
#include <snap/eb/redistribution.hpp>

namespace {
using snap::EBGeometry;
using snap::WeightedStateRedistribution;

EBGeometry line(const torch::Tensor &phi) {
  auto xf = torch::arange(phi.numel(), torch::kFloat64);
  auto transverse = torch::tensor({0., 1.}, torch::kFloat64);
  return EBGeometry::from_level_set(phi.expand({2, 2, -1}), xf, transverse,
                                    transverse);
}

TEST(WeightedStateRedistribution,
     stabilizes_sliver_and_preserves_component_budgets) {
  auto xf = torch::arange(9, torch::kFloat64);
  auto g = line(xf - (2. - 1e-7));
  auto vf = g.volume_fraction;
  WeightedStateRedistribution srd(g, torch::ones_like(vf));
  // Unit extensive provisional update in a sliver of volume 1e-7.
  auto q = torch::zeros_like(vf);
  q.index_put_({0, 0, 1}, 1. / vf.index({0, 0, 1}).item<double>());
  auto state = torch::stack({q, 2. * q, 3. * q + 4.});
  auto out = srd.apply(state);
  EXPECT_LT(out[0].abs().max().item<double>(), 10.);
  EXPECT_TRUE(torch::allclose((out * vf).sum({1, 2, 3}),
                              (state * vf).sum({1, 2, 3}), 1e-13, 1e-13));
  EXPECT_TRUE(torch::allclose(out[0] + out[1] + 4., out[2], 1e-13, 1e-13));
  EXPECT_TRUE(torch::equal(out.slice(-1, 3), state.slice(-1, 3)));
  EXPECT_TRUE(torch::equal(out.select(-1, 0), state.select(-1, 0)));
}

TEST(WeightedStateRedistribution, exactly_reconstructs_affine_cell_averages) {
  auto f = torch::arange(9, torch::kFloat64);
  auto z = torch::tensor({0., 1.}, torch::kFloat64);
  auto phi = f.view({1, 1, -1}) + 0.35 * f.view({1, -1, 1}) - 3.76;
  auto g = EBGeometry::from_level_set(phi.expand({2, 9, 9}), f, f, z);
  WeightedStateRedistribution srd(g, torch::ones_like(g.volume_fraction));
  auto q = 1.7 + 2. * g.cell_centroid[0] - 1.3 * g.cell_centroid[1];
  auto state = torch::stack({torch::ones_like(q), q});
  EXPECT_TRUE(torch::allclose(srd.apply(state), state, 1e-12, 1e-12));
}

TEST(WeightedStateRedistribution, endpoint_weights_stabilize_smoothly) {
  auto f = torch::arange(7, torch::kFloat64);
  auto impulse = torch::zeros({1, 1, 6}, torch::kFloat64);
  impulse.index_put_({0, 0, 0}, 1.);
  auto perturbation = [&](double vf) {
    auto g = line(f - (1. - vf));
    WeightedStateRedistribution srd(g, torch::ones_like(g.volume_fraction));
    return (srd.apply(impulse) - impulse).abs().max().item<double>();
  };
  EXPECT_EQ(perturbation(1.), 0.);
  EXPECT_EQ(perturbation(0.51), 0.);
  EXPECT_LT(perturbation(0.5 - 1e-5), 5e-5);
  EXPECT_GT(perturbation(0.1), 0.1);
}

TEST(WeightedStateRedistribution, periodic_seam_matches_shifted_geometry) {
  // Fluid interval crosses the periodic seam. Translating by three cells
  // moves the same pair of cut cells and their merging neighborhoods inside.
  auto phi =
      torch::tensor({1., 1., 1., -4., -4., -4., -4., -4., 1.}, torch::kFloat64);
  auto shifted = phi.slice(0, 0, 8).roll({3}, {0});
  auto g = line(phi);
  auto h = line(torch::cat({shifted, shifted.slice(0, 0, 1)}));
  auto q = torch::tensor({1., 3., -7., 2., 5., 8., 4., -5.}, torch::kFloat64)
               .reshape({1, 1, 8});
  auto run = [&](EBGeometry const &geometry, torch::Tensor state) {
    WeightedStateRedistribution srd(geometry, torch::ones_like(state),
                                    {true, false, false}, {8., 0., 0.});
    return srd.apply(state);
  };
  auto original = run(g, q);
  EXPECT_FALSE(torch::allclose(original, q));
  EXPECT_TRUE(torch::allclose(original.roll({3}, {-1}),
                              run(h, q.roll({3}, {-1})), 1e-12, 1e-12));
}

TEST(WeightedStateRedistribution, isolated_sliver_reports_setup_failure) {
  auto g = line(torch::tensor({-9., 1., -9.}, torch::kFloat64));
  EXPECT_THROW(
      WeightedStateRedistribution(g, torch::ones_like(g.volume_fraction)),
      c10::Error);
}
TEST(WeightedStateRedistribution,
     three_overlapping_neighborhoods_have_exact_weights) {
  // A quarter-cell-high fluid strip leaves no independent y data for slopes.
  // The x neighborhoods are (0,1), (1,2), (2,3), (3,2). Their overlap counts
  // are 1,2,3,2, with beta=1. An impulse in cell 2 gives Q=(0,.4,.4,.4).
  auto x = torch::arange(5, torch::kFloat64);
  auto y = torch::arange(3, torch::kFloat64);
  auto z = torch::arange(2, torch::kFloat64);
  auto phi = (y - 1.75).reshape({1, 3, 1}).expand({2, 3, 5});
  auto g = EBGeometry::from_level_set(phi, x, y, z);
  WeightedStateRedistribution srd(g, torch::ones_like(g.volume_fraction));
  auto u = torch::zeros_like(g.volume_fraction);
  u.index_put_({0, 1, 2}, 1.);
  auto expected = torch::tensor({0., .2, .4, .4}, torch::kFloat64);
  EXPECT_TRUE(torch::allclose(srd.apply(u)[0][1], expected, 1e-12, 1e-12));
}

TEST(WeightedStateRedistribution,
     member_limiter_preserves_physical_energy_and_mass) {
  auto g = line(torch::arange(6, torch::kFloat64) - 0.8);
  auto vf = g.volume_fraction;
  WeightedStateRedistribution srd(g, torch::ones_like(vf));
  auto u = torch::zeros({7, 1, 1, 5}, torch::kFloat64);
  u[0].fill_(0.5);
  u[5].fill_(0.5);
  u[6].fill_(10.);  // a tracer contributes no hydro inertia
  u[1].copy_((2. - g.cell_centroid[0]).clamp_min(-0.5));
  u[4].fill_(-9.5);  // thermal energy = 0.5 before kinetic energy subtraction
  auto minima = torch::tensor({0., -20.}, torch::kFloat64);
  auto margin = [](torch::Tensor v) {
    return v[4] + 20. * v[5] - 0.5 * v[1].square() / (v[0] + v[5]);
  };
  EXPECT_LT(margin(srd.apply(u)).min().item<double>(), 0.);
  torch::Tensor bad;
  auto out = srd.apply(u, 6, &bad, minima);
  EXPECT_FALSE(bad.item<bool>());
  EXPECT_GT(margin(out).min().item<double>(), 0.);
  EXPECT_TRUE(torch::allclose((out * vf).sum({1, 2, 3}),
                              (u * vf).sum({1, 2, 3}), 1e-13, 1e-13));
  // An inadmissible mean cannot be repaired by scaling its own slope.
  u[4].fill_(-11.);
  srd.apply(u, 6, &bad, minima);
  EXPECT_TRUE(bad.item<bool>());
}

TEST(WeightedStateRedistribution,
     member_limiter_keeps_species_and_tracers_nonnegative) {
  auto g = line(torch::arange(6, torch::kFloat64) - 0.8);
  WeightedStateRedistribution srd(g, torch::ones_like(g.volume_fraction));
  auto u = torch::zeros({7, 1, 1, 5}, torch::kFloat64);
  u[0].fill_(1.);
  u[4].fill_(1.);
  u[5].copy_(g.cell_centroid[0] - 1.);
  u[6].copy_(2. * u[5]);
  EXPECT_LT(srd.apply(u)[5].min().item<double>(), 0.);
  torch::Tensor bad;
  auto out = srd.apply(u, 6, &bad);
  EXPECT_FALSE(bad.item<bool>());
  EXPECT_GE(out[5].min().item<double>(), 0.);
  EXPECT_TRUE(torch::allclose(out[6], 2. * out[5], 1e-13, 1e-13));
}
}  // namespace
