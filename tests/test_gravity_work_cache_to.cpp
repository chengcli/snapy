// external
#include <gtest/gtest.h>

// torch
#include <torch/torch.h>

// snap
#include <snap/hydro/gravity_work_radial.hpp>

using namespace snap;

// #291: the gravity work (HydroImpl::_pe_work) and the implicit solve
// (ImplicitHydroImpl::_rx_work) cache the grid-only factors of the
// corrected-PE work, keyed on the x1f/x1v tensors and their versions.
// Module::to converts its buffers with set_data, which keeps their identity
// and version, so after a block is moved the key still matches. The cached
// factors must follow the new dtype: with the grid converted from float64 to
// float32, a cached call returns float32, bitwise equal to an uncached call
// on the converted grid. RED with a cache that checks the key only (its
// float64 factors give a float64 work).

namespace {

// x1f and x1v as a block's coordinate keeps them: buffers of a Module
struct Grid : torch::nn::Module {
  torch::Tensor x1f, x1v;
  explicit Grid(int n) {
    auto f = torch::linspace(1., 2., n + 1, torch::kFloat64);
    auto rm = f.narrow(0, 0, n), rp = f.narrow(0, 1, n);
    x1f = register_buffer("x1f", f);
    // the r^2 centroid
    x1v = register_buffer(
        "x1v", .75 * (rp.pow(4) - rm.pow(4)) / (rp.pow(3) - rm.pow(3)));
  }
};

// seam: the gravity work's path across x1 seams, corrected_pe_work_seam;
// otherwise corrected_pe_work (the whole-column gravity work and the
// implicit solve)
void check_cache_follows_to(X1Measure measure, bool seam) {
  int n = 12, is = 2, ie = 10;
  double grav1 = -9.8;
  auto grid = std::make_shared<Grid>(n);
  torch::manual_seed(291);
  auto q64 = torch::rand({3, n}, torch::kFloat64);
  auto work = [&](torch::Tensor const& q, CorrectedPEFactors* cache) {
    if (seam)
      return corrected_pe_work_seam(q, grid->x1f, grid->x1v, is, ie, grav1,
                                    measure, true, true, cache);
    return corrected_pe_work(q.slice(-1, is, ie), grid->x1f, grid->x1v, is, ie,
                             grav1, measure, cache);
  };

  CorrectedPEFactors cache;
  ASSERT_EQ(work(q64, &cache).scalar_type(), torch::kFloat64);
  auto x1f = grid->x1f;
  int64_t version = X1GridKey::version(x1f);
  grid->to(torch::kFloat32);
  // the case the dtype check is for: the key alone does not see the move
  ASSERT_TRUE(grid->x1f.is_same(x1f));
  ASSERT_EQ(X1GridKey::version(grid->x1f), version);
  ASSERT_EQ(grid->x1f.scalar_type(), torch::kFloat32);

  auto q32 = q64.to(torch::kFloat32);
  auto cached = work(q32, &cache);
  auto fresh = work(q32, nullptr);
  ASSERT_EQ(fresh.scalar_type(), torch::kFloat32);
  ASSERT_EQ(cached.scalar_type(), torch::kFloat32);
  EXPECT_TRUE(torch::equal(cached, fresh));
}

}  // namespace

TEST(GravityWorkCacheTo, column_plain) {
  check_cache_follows_to(X1Measure::plain, false);
}
TEST(GravityWorkCacheTo, column_radial) {
  check_cache_follows_to(X1Measure::radial, false);
}
TEST(GravityWorkCacheTo, column_radial_mid) {
  check_cache_follows_to(X1Measure::radial_mid, false);
}
TEST(GravityWorkCacheTo, seam_radial) {
  check_cache_follows_to(X1Measure::radial, true);
}
TEST(GravityWorkCacheTo, seam_radial_mid) {
  check_cache_follows_to(X1Measure::radial_mid, true);
}

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
