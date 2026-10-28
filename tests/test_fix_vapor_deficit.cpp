// C/C++
#include <cstdio>
#include <limits>
#include <vector>

// external
#include <gtest/gtest.h>

// torch
#include <torch/torch.h>

// snap
#include <snap/snap.h>

#include <snap/hydro/flux_positivity.hpp>
#include <snap/mesh/meshblock.hpp>

using namespace snap;

namespace {

// bottom to top: a moist layer under one slightly negative cell and two dry
// ones, the shape of the TJ16 species trigger (-2.5e-11 above moist air)
std::vector<double> const kColumn = {2.e-3, 1.5e-3, 1.e-3, -2.5e-11,
                                     1.e-16, 1.e-16};
double const kDry = 0.8;  // dry density of every cell

struct Repaired {
  std::vector<double> vapor;
  bool marked;
};

//! one conserved-limiter call on the column with cell 3 set to `negative`
Repaired repair(double negative) {
  static auto block = [] {
    auto options =
        MeshBlockOptionsImpl::from_yaml("test_fix_vapor_deficit.yaml");
    auto b = std::make_shared<MeshBlockImpl>(options);
    b->to(torch::kCPU, torch::kFloat64);
    return b;
  }();
  auto coord = block->pcoord;
  auto peos = block->phydro->peos;
  int il = coord->il();
  auto cons = torch::zeros({peos->nvar(), coord->options->nc3(),
                            coord->options->nc2(), coord->options->nc1()},
                           torch::kFloat64);
  cons[IDN].fill_(kDry);
  cons[IPR].fill_(2.e5);
  for (int k = 0; k < kColumn.size(); ++k) {
    cons[ICY].select(-1, il + k).fill_(k == 3 ? negative : kColumn[k]);
  }
  // ghosts hold the edge cells, as a reflecting wall leaves them
  for (int g = 0; g < il; ++g) {
    cons[ICY].select(-1, g).fill_(kColumn.front());
    cons[ICY].select(-1, il + kColumn.size() + g).fill_(kColumn.back());
  }

  peos->reset_limiter_marks(cons);
  peos->apply_conserved_limiter_(cons);

  Repaired out;
  for (int k = 0; k < kColumn.size(); ++k) {
    out.vapor.push_back(cons[ICY].select(-1, il + k).item<double>());
  }
  out.marked = peos->limiter_marks()[0].item<bool>();
  return out;
}

}  // namespace

// The vapor column repair borrows only the deficit: the negative cell goes to
// zero, and the cells below that cover it give up their proportional share of
// it, nothing more. The re-mix it replaces moved the moist cell under the
// negative one by ~5e-4 for a 2.5e-11 deficit, and so marked a redo that no
// smaller dt could clear.
TEST(eos, fix_vapor_borrows_only_the_deficit) {
  double deficit = -kColumn[3];
  auto r = repair(kColumn[3]);

  double before = 0., after = 0.;
  for (int k = 0; k < kColumn.size(); ++k) {
    std::printf("cell %d: %.17g -> %.17g (moved %.6g)\n", k, kColumn[k],
                r.vapor[k], r.vapor[k] - kColumn[k]);
    EXPECT_GE(r.vapor[k], 0.) << "cell " << k;
    before += kColumn[k];
    after += r.vapor[k];
  }
  std::printf("column mass: relative change %.3g, marked %d\n",
              (after - before) / before, int(r.marked));
  EXPECT_NEAR((after - before) / before, 0., 1.e-15);

  // the donors are the cells below the negative one down to the first that
  // covers it -- here cell 2 alone, which gives the whole deficit; the cells
  // under it and the dry cells above keep their values bit for bit
  double donors = kColumn[2];
  EXPECT_NEAR(kColumn[2] - r.vapor[2], deficit * kColumn[2] / donors,
              4. * std::numeric_limits<double>::epsilon() * kColumn[2]);
  EXPECT_EQ(r.vapor[0], kColumn[0]);
  EXPECT_EQ(r.vapor[1], kColumn[1]);
  EXPECT_EQ(r.vapor[3], 0.);
  EXPECT_EQ(r.vapor[4], kColumn[4]);
  EXPECT_EQ(r.vapor[5], kColumn[5]);

  // 2.5e-11 is above the mark (4096 ulp of the cell's ~0.8 kg/m3): a redo.
  // kSpeciesRepairMarkUlp defaults to the positivity round-off; a test that
  // runs on main as well reads that one
  double mark = positivity_roundoff(torch::kFloat64) * kDry;
  ASSERT_GT(deficit, mark);
  EXPECT_TRUE(r.marked);
}

// The mark follows the deficit, not the donors' change: a deficit below
// kSpeciesRepairMarkUlp of the gas density is repaired without a redo, though
// it sits on the same moist cell.
TEST(eos, fix_vapor_marks_on_the_deficit) {
  double mark = positivity_roundoff(torch::kFloat64) * kDry;
  double small = 0.25 * mark;
  auto r = repair(-small);
  std::printf("deficit %.3g (mark %.3g): cell 2 moved %.6g, marked %d\n",
              small, mark, r.vapor[2] - kColumn[2], int(r.marked));
  EXPECT_EQ(r.vapor[3], 0.);
  EXPECT_NEAR(kColumn[2] - r.vapor[2], small,
              4. * std::numeric_limits<double>::epsilon() * kColumn[2]);
  EXPECT_FALSE(r.marked);
}
