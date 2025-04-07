// C/C++
#include <sstream>
#include <stdexcept>

// snap
#include "oct_tree.hpp"

namespace snap {
torch::optional<OctTreeNode> OctTreeImpl::find_neighbor(LogicalLocation myloc,
                                                        NeighborIndex nx,
                                                        BoundaryFlag const *bcs,
                                                        bool amrflag) {
  std::stringstream msg;

  int ox, oy, oz;
  int64_t lx = myloc->lx1;
  int64_t ly = myloc->lx2;
  int64_t lz = myloc->lx3;
  int ll = myloc->level;
  int level_diff = ll - root_level();

  lx += nx.ox1;
  ly += nx.ox2;
  lz += nx.ox3;

  // periodic and polar boundaries
  if (lx < 0) {
    if (bcs[BoundaryFace::kInnerX1] == BoundaryFlag::kPeriodic ||
        bcs[BoundaryFace::kInnerX1] == BoundaryFlag::kShearPeriodic)
      lx = (options.nb1() << level_diff) - 1;
    else
      return torch::nullopt;
  }

  if (lx >= options.nb1() << level_diff) {
    if (bcs[BoundaryFace::kOuterX1] == BoundaryFlag::kPeriodic ||
        bcs[BoundaryFace::kOuterX1] == BoundaryFlag::kShearPeriodic)
      lx = 0;
    else
      return torch::nullopt;
  }

  bool polar = false;

  if (ly < 0) {
    if (bcs[BoundaryFace::kInnerX2] == BoundaryFlag::kPeriodic) {
      ly = (options.nb2() << level_diff) - 1;
    } else if (bcs[BoundaryFace::kInnerX2] == BoundaryFlag::kPolar) {
      ly = 0;
      polar = true;
    } else {
      return torch::nullopt;
    }
  }

  if (ly >= options.nb2() << level_diff) {
    if (bcs[BoundaryFace::kOuterX2] == BoundaryFlag::kPeriodic) {
      ly = 0;
    } else if (bcs[BoundaryFace::kOuterX2] == BoundaryFlag::kPolar) {
      ly = (options.nb2() << level_diff) - 1;
      polar = true;
    } else {
      return torch::nullopt;
    }
  }

  std::int64_t num_x3 = options.nb3() << level_diff;

  if (lz < 0) {
    if (bcs[BoundaryFace::kInnerX3] == BoundaryFlag::kPeriodic)
      lz = num_x3 - 1;
    else
      return torch::nullopt;
  }

  if (lz >= num_x3) {
    if (bcs[BoundaryFace::kOuterX3] == BoundaryFlag::kPeriodic)
      lz = 0;
    else
      return torch::nullopt;
  }

  if (ll < 1) return root;  // single grid; return root
  if (polar) lz = (lz + num_x3 / 2) % num_x3;

  auto bt = root;
  for (int level = 0; level < ll; level++) {
    if (bt->leaves.size() == 0) {  // leaf
      if (level == ll - 1) {
        return bt;
      } else {
        msg << "### FATAL ERROR in FindNeighbor" << std::endl
            << "Neighbor search failed. The Block Tree is broken." << std::endl;
        throw std::runtime_error(msg.str());
      }
    }

    // find a leaf in the next level
    int sh = ll - level - 1;
    ox = ((lx >> sh) & 1LL) == 1LL;
    oy = ((ly >> sh) & 1LL) == 1LL;
    oz = ((lz >> sh) & 1LL) == 1LL;
    if (bt->leaf(ox, oy, oz).is_empty()) {
      msg << "### FATAL ERROR in FindNeighbor" << std::endl
          << "Neighbor search failed. The Block Tree is broken." << std::endl;
      throw std::runtime_error(msg.str());
    }

    bt = bt->leaf(ox, oy, oz).get<OctTreeNode>();
  }

  if (bt->leaves.size() == 0)  // leaf on the same level
    return bt;

  // one level finer: check if it is a leaf
  ox = oy = oz = 0;
  if (nx.ox1 < 0) ox = 1;
  if (nx.ox2 < 0) oy = 1;
  if (nx.ox3 < 0) oz = 1;

  if (bt->leaf(ox, oy, oz).is_empty()) {
    msg << "### FATAL ERROR in FindNeighbor" << std::endl
        << "Neighbor search failed. The Block Tree is broken." << std::endl;
    throw std::runtime_error(msg.str());
  }

  auto btleaf = bt->leaf(ox, oy, oz).get<OctTreeNode>();

  if (btleaf->leaves.size() == 0) return bt;  // return this block

  if (!amrflag) {
    msg << "### FATAL ERROR in FindNeighbor" << std::endl
        << "Neighbor search failed. The Block Tree is broken." << std::endl;
    throw std::runtime_error(msg.str());
  }

  return bt;
}
}  // namespace snap
