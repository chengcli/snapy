// RED on main 3f7ad96. One process, the same column as one block and as two
// blocks. p_ref of the cell under the seam should be one value. It is not.
//
// Two interior cells is too small: a block whose nx1 is 1 stores no ghosts
// (CoordinateOptions::nc1() is 1, and il() is 0). p_ref then takes the edge
// stencil instead of the hydrostatic face. Four interior cells, two per
// block, is the smallest split where that cell is an ordinary interior cell.
//
// HydroImpl passes the running face down the column only when the split is
// across processes (hydro.cpp, the x1 relay). A second block in the same
// process gets an empty anchor, and the scan starts again from that block's
// own top cell (hydro_ref_x1_scan_impl, anchor == nullptr). This is the p_ref
// seam jump reported on #250. The harness that measured it is
// chengcli/snapy study/250-t4-harness at
// 1ffb3c16d8c6e182461876a30734af73320a45a9. This test does not fix it.

// external
#include <gtest/gtest.h>

// C/C++
#include <cmath>
#include <tuple>
#include <utility>

// torch
#include <torch/torch.h>

// snap
#include <snap/snap.h>

#include <snap/hydro/hydro.hpp>
#include <snap/mesh/mesh.hpp>

using namespace snap;

namespace {

// _hydro_ref_x1 is protected. A derived class may name it; the member
// pointer applies to any HydroImpl. Same access as the T4 harness.
struct RefPeek : HydroImpl {
  static auto ptr() { return &RefPeek::_hydro_ref_x1; }
};

// Return is (psf_lo, pref, dsf, dref). The seam check uses pref.
torch::Tensor pref_of(Hydro const& hydro, torch::Tensor const& w) {
  return std::get<1>(((*hydro).*RefPeek::ptr())(w));
}

Mesh make_mesh(int nb1) {
  auto block_opts =
      MeshBlockOptionsImpl::from_yaml("test_pref_local_seam.yaml");
  if (nb1 > 1) {
    block_opts->layout()->type() = "cubed";
    block_opts->layout()->pz(nb1);
  }
  auto mesh_opts = MeshOptionsImpl::create();
  mesh_opts->block(block_opts);
  mesh_opts->blocks_per_process(nb1);
  return Mesh(mesh_opts);
}

torch::Tensor uniform_w(MeshBlock block) {
  auto coord = block->pcoord;
  auto w = torch::zeros({block->phydro->peos->nvar(), coord->options->nc3(),
                         coord->options->nc2(), coord->options->nc1()},
                        torch::kFloat64);
  w[IDN].fill_(1.);
  w[IPR].fill_(1.e5);
  return w;
}

double at_cell(torch::Tensor const& field, int i) {
  return field.select(-1, i).item<double>();
}

// Cell-center p_ref if the scan restarts on this block. The anchor is this
// cell's own top face, then half a cell of g*rho*dx. grav1 is stored < 0.
double restarted_pref(MeshBlock block, torch::Tensor const& w) {
  auto coord = block->pcoord;
  int iu = coord->iu();
  double g = -block->phydro->options->grav()->grav1();
  double dx = coord->dx1f.select(-1, iu).item<double>();
  double pres = w[IPR].select(-1, iu).item<double>();
  double rho = w[IDN].select(-1, iu).item<double>();
  double anchor = pres * std::exp(-g * 0.5 * dx / (pres / rho));
  return anchor + 0.5 * g * rho * dx;
}

}  // namespace

// Uniform rho = 1 and p = 1e5, one process, dx = 1. p_ref of the cell under
// the seam is fixed by the cells above it. Splitting the column inside one
// process must not change it.
TEST(HydroRefX1, local_blocks_restart_the_reference_at_the_seam) {
  auto one = make_mesh(1);
  ASSERT_EQ(one->blocks.size(), 1u);
  auto block = one->blocks[0];
  auto w = uniform_w(block);
  auto pref = pref_of(block->phydro, w);
  int il = block->pcoord->il();
  int iu = block->pcoord->iu();
  int nint = iu - il + 1;
  ASSERT_GE(nint, 4);
  ASSERT_EQ(nint % 2, 0);
  int below = il + nint / 2 - 1;  // interior cell just under the mid seam
  int above = below + 1;
  double unsplit_below = at_cell(pref, below);
  double unsplit_above = at_cell(pref, above);

  auto two = make_mesh(2);
  ASSERT_EQ(two->blocks.size(), 2u);
  MeshBlock lower = two->blocks[0];
  MeshBlock upper = two->blocks[1];
  if (lower->pcoord->options->x1min() > upper->pcoord->options->x1min())
    std::swap(lower, upper);
  ASSERT_LT(lower->pcoord->options->x1min(), upper->pcoord->options->x1min());
  ASSERT_EQ(lower->pcoord->iu() - lower->pcoord->il() + 1, nint / 2);
  ASSERT_EQ(upper->pcoord->iu() - upper->pcoord->il() + 1, nint / 2);

  auto wu = uniform_w(upper);
  auto wl = uniform_w(lower);
  double from_above = at_cell(pref_of(upper->phydro, wu), upper->pcoord->il());
  double from_below = at_cell(pref_of(lower->phydro, wl), lower->pcoord->iu());

  // The block that still owns the domain top reproduces the unsplit cell.
  EXPECT_NEAR(from_above, unsplit_above, 1e-6)
      << "upper block p_ref " << from_above << " unsplit " << unsplit_above;

  // RED. The lower block does not. Its scan started again from its own top
  // cell, so this p_ref is not the unsplit value.
  EXPECT_NEAR(from_below, unsplit_below, 1e-6)
      << "lower block p_ref " << from_below << " unsplit " << unsplit_below
      << " restart-from-own-top " << restarted_pref(lower, wl);
}
