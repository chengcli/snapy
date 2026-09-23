// C/C++
#include <cmath>
#include <vector>

// gtest
#include <gtest/gtest.h>

// torch
#include <torch/torch.h>

// snap
#include <snap/snap.h>

#include <snap/hydro/balance_column.hpp>
#include <snap/hydro/hydro_dispatch.hpp>

namespace {

constexpr double kGrav = 10.44;  // Saturn-like g; kRd is a Jovian H2/He gas
constexpr double kRd = 3777.0;
constexpr double kCp = 3.5 * kRd;
constexpr double kTs = 300.0;
constexpr double kPs = 1.0e5;

struct Column {
  torch::Tensor w;     // (nvar, 1, 1, nx1), ghost-free
  torch::Tensor dx1f;  // (nx1,)
};

//! A dry adiabat integrated CELL BY CELL -- the IC builders' own march, and so
//! the defect: exact for the continuum ODE to O(dz^2), off the scheme's
//! discrete balance by that truncation.
Column marched_column(int nx1, double height, bool uniform) {
  auto opt = torch::TensorOptions().dtype(torch::kFloat64);
  auto dx1f = torch::empty({nx1}, opt);
  auto a = dx1f.accessor<double, 1>();
  double dz = height / nx1;
  for (int i = 0; i < nx1; ++i) {
    // a smoothly stretched grid, so the non-uniform branch is a real case
    a[i] = uniform || nx1 < 2 ? dz : dz * (0.6 + 0.8 * i / double(nx1 - 1));
  }

  auto w = torch::zeros({snap::IPR + 1, 1, 1, nx1}, opt);
  auto rho_t = w[snap::IDN],
       prs_t = w[snap::IPR];  // accessor() is deleted on an rvalue
  auto rho = rho_t.accessor<double, 3>();
  auto prs = prs_t.accessor<double, 3>();
  double z = 0., p = kPs;
  for (int i = 0; i < nx1; ++i) {
    double zc = z + 0.5 * a[i];
    double t = kTs - kGrav * zc / kCp;
    // forward Euler from the previous centre: the truncation this removes
    if (i > 0) {
      double zp = z - 0.5 * a[i - 1];
      double tp = kTs - kGrav * zp / kCp;
      p -= kGrav * (p / (kRd * tp)) * (zc - zp);
    } else {
      p -= kGrav * (kPs / (kRd * kTs)) * zc;
    }
    prs[0][0][i] = p;
    rho[0][0][i] = p / (kRd * t);
    z += a[i];
  }
  // a tracer-free velocity field the balance must carry through untouched
  w[snap::IVX].fill_(0.25);
  w[snap::IVY].fill_(-0.5);
  return {w, dx1f};
}

struct Ref {
  torch::Tensor psf_lo, psf_hi, pref, dsf, dref;
};

Ref reference(torch::Tensor const& w, torch::Tensor const& dx1f, int iu,
              bool uniform, bool phys_in, bool phys_out, bool wall_clamp) {
  auto sizes = w.sizes().slice(1).vec();
  Ref r{torch::empty(sizes, w.options()), torch::empty(sizes, w.options()),
        torch::empty(sizes, w.options()), torch::empty(sizes, w.options()),
        torch::empty(sizes, w.options())};
  torch::Tensor anchor;
  at::native::call_hydro_ref_x1(w.device().type(), w.contiguous(),
                                dx1f.contiguous(), anchor, r.psf_lo, r.psf_hi,
                                r.pref, r.dsf, r.dref, iu, kGrav, uniform,
                                phys_in, phys_out, wall_clamp);
  return r;
}

//! The same column inside a block: `ng` ghost cells at each end, filled with
//! the even mirror a reflecting wall writes and then moved, so anything that
//! reads a ghost reads a wrong number.
Column padded(Column const& c, int ng) {
  int nx1 = c.w.size(-1);
  int nc1 = nx1 + 2 * ng;
  auto w = torch::zeros({c.w.size(0), 1, 1, nc1}, c.w.options());
  w.narrow(-1, ng, nx1).copy_(c.w);
  auto dx1f = torch::empty({nc1}, c.dx1f.options());
  dx1f.narrow(-1, ng, nx1).copy_(c.dx1f);
  for (int m = 0; m < ng; ++m) {
    w.narrow(-1, ng - 1 - m, 1).copy_(1.3 * c.w.narrow(-1, m, 1));
    w.narrow(-1, ng + nx1 + m, 1).copy_(1.3 * c.w.narrow(-1, nx1 - 1 - m, 1));
    dx1f.narrow(-1, ng - 1 - m, 1).copy_(c.dx1f.narrow(-1, m, 1));
    dx1f.narrow(-1, ng + nx1 + m, 1).copy_(c.dx1f.narrow(-1, nx1 - 1 - m, 1));
  }
  return {w, dx1f};
}

//! max |p' - p'(top)| / (rho g dz): the acceleration the column can still feel,
//! in units of g. This re-runs the SAME kernel balance_column iterates against,
//! so it audits the fixed point, not the reference -- the reference is
//! test_hydro_ref_x1's job. What it does catch is the whole class of ways the
//! iteration can report convergence against something other than what it
//! returned: a stale residual, a gauge read at the wrong cell, a `uniform`
//! flag decided differently from the solver's rule.
double residual(Column const& c, bool uniform) {
  int nx1 = c.w.size(-1);
  auto r = reference(c.w, c.dx1f, nx1 - 1, uniform, true, true, true);
  auto pp = c.w[snap::IPR] - r.pref;
  auto gauge = pp.narrow(-1, nx1 - 1, 1);
  return ((pp - gauge).abs() / (c.w[snap::IDN] * kGrav * c.dx1f))
      .max()
      .item<double>();
}

// THE PROPERTY THE WHOLE DESIGN RESTS ON. A ghost-free column and the same
// column sitting inside a block with both x1 walls physical must produce the
// SAME interior reference -- otherwise a balance built on the column outside
// the solver is not the balance the solver enforces inside it. It holds
// because the clamp's one-sided rows and the face scan read owned cells only.
TEST(BalanceColumn, a_ghost_free_column_reproduces_a_blocks_own_reference) {
  for (bool uniform : {true, false}) {
    auto c = marched_column(64, 3.0e5, uniform);
    auto b = padded(c, 3);
    int nx1 = c.w.size(-1);

    auto free_ref = reference(c.w, c.dx1f, nx1 - 1, uniform, true, true, true);
    auto blk_ref =
        reference(b.w, b.dx1f, 3 + nx1 - 1, uniform, true, true, true);

    EXPECT_TRUE(torch::equal(free_ref.pref, blk_ref.pref.narrow(-1, 3, nx1)))
        << "uniform=" << uniform;
    EXPECT_TRUE(
        torch::equal(free_ref.psf_lo, blk_ref.psf_lo.narrow(-1, 3, nx1)))
        << "uniform=" << uniform;
    EXPECT_TRUE(
        torch::equal(free_ref.psf_hi, blk_ref.psf_hi.narrow(-1, 3, nx1)))
        << "uniform=" << uniform;
    EXPECT_TRUE(torch::equal(free_ref.dref, blk_ref.dref.narrow(-1, 3, nx1)))
        << "uniform=" << uniform;

    // dsf is the ONE reference output that does not transfer, and only at the
    // bottom-most cell: its one-sided fallback keys on the absolute index i > 0
    // rather than on i > il, so a ghost-free column takes rop_smooth(il) where
    // a block averages it with rop_smooth(il-1). Both read owned cells only --
    // the clamp is not leaking -- they weight them differently. dsf plays no
    // part in the fixed point, which reads pref alone, so the balance is
    // unaffected. Asserted rather than omitted, so the exception stays visible
    // if it moves.
    EXPECT_TRUE(torch::equal(free_ref.dsf.narrow(-1, 1, nx1 - 1),
                             blk_ref.dsf.narrow(-1, 4, nx1 - 1)))
        << "uniform=" << uniform;
    EXPECT_FALSE(torch::equal(free_ref.dsf.narrow(-1, 0, 1),
                              blk_ref.dsf.narrow(-1, 3, 1)))
        << "uniform=" << uniform;
  }
}

// ...and the control, so the test above cannot pass on a build where padded and
// ghost-free agree whatever the clamp does. With the clamp OFF they diverge,
// and STRUCTURALLY, not by ghost contamination: the padded wall cell il+1
// satisfies the interior guard i >= 2 and takes the six-face stencil, while the
// ghost-free cell 1 fails it and keeps the two-point mean. (That ghosts leak at
// all with the clamp off is test_hydro_ref_x1's control, measured there
// directly.)
TEST(BalanceColumn, without_the_clamp_the_two_references_disagree) {
  auto c = marched_column(64, 3.0e5, /*uniform=*/true);
  auto b = padded(c, 3);
  int nx1 = c.w.size(-1);

  auto free_ref = reference(c.w, c.dx1f, nx1 - 1, true, true, true, false);
  auto blk_ref = reference(b.w, b.dx1f, 3 + nx1 - 1, true, true, true, false);

  EXPECT_FALSE(torch::equal(free_ref.pref, blk_ref.pref.narrow(-1, 3, nx1)));
}

TEST(BalanceColumn, a_marched_column_comes_out_at_rest) {
  for (bool uniform : {true, false}) {
    auto c = marched_column(64, 3.0e5, uniform);
    double before = residual(c, uniform);
    EXPECT_GT(before, 1.e-4)
        << "the fixture is not the defect: uniform=" << uniform;

    auto [wb, err, sweeps] = snap::balance_column(c.w, c.dx1f, kGrav);
    Column balanced{wb, c.dx1f};
    // `err < rtol` is the function's own break condition, so asserting it here
    // could not fail. What CAN fail, and is what the header promises, is that
    // the number it returns describes the state it returned -- recompute it
    // independently and require the two to agree.
    EXPECT_DOUBLE_EQ(residual(balanced, uniform), err) << "uniform=" << uniform;
    EXPECT_GT(sweeps, 0);
  }
}

// p/rho pins the temperature for the ideal mixtures this is for, so the whole
// point is that only p and rho move, and together.
TEST(BalanceColumn, the_temperature_and_every_other_channel_stay_put) {
  auto c = marched_column(64, 3.0e5, /*uniform=*/true);
  auto rt0 = c.w[snap::IPR] / c.w[snap::IDN];

  auto [wb, err, sweeps] = snap::balance_column(c.w, c.dx1f, kGrav);
  auto rt1 = wb[snap::IPR] / wb[snap::IDN];

  EXPECT_LT(((rt1 - rt0).abs() / rt0).max().item<double>(), 1.e-14);
  EXPECT_TRUE(torch::equal(wb[snap::IVX], c.w[snap::IVX]));
  EXPECT_TRUE(torch::equal(wb[snap::IVY], c.w[snap::IVY]));
  // and the column really did move -- otherwise the line above is vacuous
  EXPECT_GT(((wb[snap::IPR] - c.w[snap::IPR]).abs() / c.w[snap::IPR])
                .max()
                .item<double>(),
            1.e-8);
  // the input is not modified in place
  EXPECT_GT(residual(c, true), 1.e-4);
}

TEST(BalanceColumn, a_balanced_column_is_a_fixed_point) {
  auto c = marched_column(64, 3.0e5, /*uniform=*/true);
  auto [w1, e1, n1] = snap::balance_column(c.w, c.dx1f, kGrav);
  auto [w2, e2, n2] = snap::balance_column(w1, c.dx1f, kGrav);

  EXPECT_EQ(n2, 0);
  EXPECT_TRUE(torch::equal(w1, w2));
}

// A block with fewer cells than it has ghosts cannot carry a one-sided row at
// all: the outer row would start at face iu-4 < 0, and hydro_ref_x1_face
// saturates ABOVE nc1 but not below, so it would read before the column -- in a
// multi-column array, that is the PREVIOUS COLUMN's top face.
//
// So the property to assert is the one such a read violates and that a test can
// actually see: columns are independent. TWO columns, thin block, and moving
// one must not move the other's reference. (Asserting "the wall cell equals the
// two-point mean" instead would not discriminate: an out-of-range value usually
// falls outside the cell's [lo, hi] bracket and is rejected BACK to that mean,
// so the broken kernel would pass.)
TEST(BalanceColumn, a_block_thinner_than_its_ghosts_keeps_its_columns_apart) {
  constexpr int nc1 = 7, iu = 3;  // il = 3, ONE interior cell: iu < 4
  auto opt = torch::TensorOptions().dtype(torch::kFloat64);
  auto make = [&](double scale) {
    auto w = torch::zeros({snap::IPR + 1, 1, 2, nc1}, opt);
    auto prs_t = w[snap::IPR], rho_t = w[snap::IDN];
    auto prs = prs_t.accessor<double, 3>();
    auto rho = rho_t.accessor<double, 3>();
    for (int c = 0; c < 2; ++c)
      for (int i = 0; i < nc1; ++i) {
        double s =
            c == 0 ? scale : 1.0;  // column 1 is the same in both fixtures
        prs[0][c][i] = s * 1.0e5 * std::exp(-0.15 * i);
        rho[0][c][i] = prs[0][c][i] / (287.0 * 300.0);
      }
    return w;
  };
  auto dx1f = torch::full({nc1}, 1.0e3, opt);
  // wall at x1-outer, SEAM at x1-inner: the arm the narrowed gate re-opened
  auto a = reference(make(1.0), dx1f, iu, true, false, true, true);
  auto b = reference(make(7.0), dx1f, iu, true, false, true, true);
  EXPECT_TRUE(torch::equal(a.pref.narrow(-2, 1, 1), b.pref.narrow(-2, 1, 1)))
      << "column 1's reference moved when column 0 did\n"
      << a.pref << "\n"
      << b.pref;
}

TEST(BalanceColumn, it_refuses_what_it_cannot_deliver) {
  auto c = marched_column(64, 3.0e5, /*uniform=*/true);
  // a column too short for the reference's own wall rows
  EXPECT_THROW(snap::balance_column(c.w.narrow(-1, 0, 4),
                                    c.dx1f.narrow(-1, 0, 4), kGrav),
               c10::Error);
  // no clamp: the reference would read outside the column at each wall
  EXPECT_THROW(snap::balance_column(c.w, c.dx1f, kGrav, /*wall_clamp=*/false),
               c10::Error);
  // a gravity sign, not a magnitude
  EXPECT_THROW(snap::balance_column(c.w, c.dx1f, -kGrav), c10::Error);
  // and an unconverged sweep budget is an error, never a quiet return
  EXPECT_THROW(
      snap::balance_column(c.w, c.dx1f, kGrav, true, 1.e-10, /*max_iter=*/1),
      c10::Error);
}

}  // namespace
