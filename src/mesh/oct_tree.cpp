//! \brief implementation of functions in the OctTree class
//!
//! The OctTree stores the logical grid structure, and is used for
//! neighbor searches, assigning global IDs, etc.  Level is defined as "logical
//! level", where the logical root (single block) level is 0.  Note the logical
//! level of the physical root grid (user-specified root grid) will be greater
//! than zero if it contains more than one MeshBlock

// C/C++ headers
#include <iostream>
#include <stdexcept>

// base
#include <configure.h>

// snap
#include "mesh_formatter.hpp"
#include "oct_tree.hpp"

namespace snap {
bool LogicalLocationImpl::lesser(const LogicalLocation &other) const {
  return level < other->level;
}

bool LogicalLocationImpl::greater(const LogicalLocation &other) const {
  return level > other->level;
}

bool LogicalLocationImpl::equal(const LogicalLocation &other) const {
  return (level == other->level) && (lx1 == other->lx1) &&
         (lx2 == other->lx2) && (lx3 == other->lx3);
}

LogicalLocationImpl::LogicalLocationImpl(int level_, int64_t lx1_, int64_t lx2_,
                                         int64_t lx3_)
    : level(level_), lx1(lx1_), lx2(lx2_), lx3(lx3_) {
  reset();
}

void LogicalLocationImpl::reset() {}

void LogicalLocationImpl::pretty_print(std::ostream &os) const {
  os << "LogicalLocation(level=" << level << ", lx1=" << lx1 << ", lx2=" << lx2
     << ", lx3=" << lx3 << ")";
}

std::string LogicalLocationImpl::to_string() const {
  std::ostringstream os;
  os << std::to_string(level) << "x" << std::to_string(lx1) << "x"
     << std::to_string(lx2) << "x" << std::to_string(lx3);
  return os.str();
}

OctTreeOptions::OctTreeOptions(ParameterInput pin) {
  int nx1 = pin->GetOrAddInteger("mesh", "nx1", 1);
  int nx2 = pin->GetOrAddInteger("mesh", "nx2", 1);
  int nx3 = pin->GetOrAddInteger("mesh", "nx3", 1);

  if (pin->DoesParameterExist("meshblock", "nx1")) {
    nb1(nx1 / pin->GetInteger("meshblock", "nx1"));
  } else {
    nb1(1);
  }

  if (pin->DoesParameterExist("meshblock", "nx2")) {
    nb2(nx2 / pin->GetInteger("meshblock", "nx2"));
  } else {
    nb2(1);
  }

  if (pin->DoesParameterExist("meshblock", "nx3")) {
    nb3(nx3 / pin->GetInteger("meshblock", "nx3"));
  } else {
    nb3(1);
  }

  ndim(1);
  if (nx2 > 1) ndim(2);
  if (nx3 > 1) ndim(3);
}

OctTreeNodeImpl::OctTreeNodeImpl(OctTreeOptions const &options_)
    : options(options_) {
  reset();
}

OctTreeNodeImpl::OctTreeNodeImpl(OctTreeOptions const &options_,
                                 LogicalLocation parent_loc, int n)
    : options(options_) {
  loc = register_module("loc", LogicalLocation());

  loc->lx1 = (parent_loc->lx1 << 1) + n & 1;
  loc->lx2 = (parent_loc->lx2 << 1) + (n >> 1) & 1;
  loc->lx3 = (parent_loc->lx3 << 1) + (n >> 2) & 1;
  loc->level = parent_loc->level + 1;
}

void OctTreeNodeImpl::reset() {
  if (loc->level == 0) {
    loc = register_module("loc", LogicalLocation());
  }

  int root_level = octree_root_level(options);
  if (loc->level == root_level) return;

  leaves.resize(nleaf());

  int64_t levfac = 1LL << (root_level - loc->level - 1);
  for (int n = 0; n < nleaf(); n++) {
    int i = n & 1;
    int j = (n >> 1) & 1;
    int k = (n >> 2) & 1;
    if ((loc->lx3 << 1 + k) * levfac < options.nb3() &&
        (loc->lx2 << 1 + j) * levfac < options.nb2() &&
        (loc->lx1 << 1 + i) * levfac < options.nb1()) {
      leaves[n] = register_module("leaf" + std::to_string(n),
                                  OctTreeNode(options, loc, n));
      leaves[n].get<OctTreeNode>()->reset();
    }
  }
  return;
}

void OctTreeNodeImpl::add_node(LogicalLocation rloc, int &nnew) {
  if (loc->level == rloc->level) return;  // done

  if (leaves.size() == 0) {  // leaf -> create the finer level
    refine(nnew);
  }

  // get leaf index
  int sh = rloc->level - loc->level - 1;
  int m1 = ((rloc->lx1 >> sh) & 1LL) == 1LL;
  int m2 = ((rloc->lx2 >> sh) & 1LL) == 1LL;
  int m3 = ((rloc->lx3 >> sh) & 1LL) == 1LL;
  int n = m1 + (m2 << 1) + (m3 << 2);

  leaves[n].get<OctTreeNode>()->add_node(rloc, nnew);

  return;
}

void OctTreeNodeImpl::add_node_without_refine(LogicalLocation rloc) {
  if (loc->level == rloc->level) return;  // done

  if (leaves.size() == 0) {
    leaves.resize(nleaf());
  }

  // get leaf index
  int sh = rloc->level - loc->level - 1;
  int m1 = ((rloc->lx1 >> sh) & 1LL) == 1LL;
  int m2 = ((rloc->lx2 >> sh) & 1LL) == 1LL;
  int m3 = ((rloc->lx3 >> sh) & 1LL) == 1LL;
  int n = m1 + (m2 << 1) + (m3 << 2);

  if (leaves[n].is_empty()) {
    leaves[n] = register_module("leaf" + std::to_string(n),
                                OctTreeNode(options, loc, n));
  }
  leaves[n].get<OctTreeNode>()->add_node_without_refine(rloc);

  return;
}

void OctTreeNodeImpl::count(int &num) {
  if (loc->level == 0) num = 0;

  if (leaves.size() == 0) {
    num++;
  } else {
    for (int n = 0; n < nleaf(); n++) {
      if (!leaves[n].is_empty()) leaves[n].get<OctTreeNode>()->count(num);
    }
  }
  return;
}

torch::optional<OctTreeNode> OctTreeNodeImpl::find_node(LogicalLocation tloc) {
  if (tloc->level == loc->level) {
    return std::static_pointer_cast<OctTreeNodeImpl>(shared_from_this());
  }
  if (leaves.size() == 0) return torch::nullopt;

  // get leaf index
  int sh = tloc->level - loc->level - 1;
  int m1 = ((tloc->lx1 >> sh) & 1LL) == 1LL;
  int m2 = ((tloc->lx2 >> sh) & 1LL) == 1LL;
  int m3 = ((tloc->lx3 >> sh) & 1LL) == 1LL;
  int n = m1 + (m2 << 1) + (m3 << 2);

  if (n > leaves.size()) {
    throw std::runtime_error("OctTreeNodeImpl::find_node: Invalid leaf index");
  } else {
    if (leaves[n].is_empty()) {
      return torch::nullopt;
    } else {
      return leaves[n].get<OctTreeNode>()->find_node(tloc);
    }
  }
}

torch::Tensor OctTreeNodeImpl::forward(std::vector<OctTreeNode> *list) {
  if (leaves.size() == 0) {
    list->push_back(
        std::static_pointer_cast<OctTreeNodeImpl>(shared_from_this()));
  } else {
    for (int n = 0; n < nleaf(); n++) {
      if (!leaves[n].is_empty()) leaves[n].forward(list);
    }
  }
  // dummpy
  return torch::tensor(0, torch::kInt);
}

OctTreeImpl::OctTreeImpl(OctTreeOptions const &options_) : options(options_) {
  reset();
}

void OctTreeImpl::reset() {
  root = register_module("root", OctTreeNode(options));
}

std::vector<OctTreeNode> OctTreeImpl::forward() {
  std::vector<OctTreeNode> list;
  root->forward(&list);
  return list;
}

int octree_root_level(OctTreeOptions const &op) {
  int nbmax = std::max(op.nb1(), std::max(op.nb2(), op.nb3()));
  int root_level;
  for (root_level = 0; (1 << root_level) < nbmax; root_level++);
  return root_level;
}

}  // namespace snap
