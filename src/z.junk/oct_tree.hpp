#pragma once

// C/C++ headers
#include <string>
#include <unordered_map>
#include <vector>

// torch
#include <torch/nn/cloneable.h>
#include <torch/nn/module.h>
#include <torch/nn/modules/common.h>
#include <torch/nn/modules/container/any.h>
#include <torch/nn/modules/container/modulelist.h>

// snap
#include <snap/bc/bc.hpp>
#include <snap/input/parameter_input.hpp>

// arg
#include <snap/add_arg.h>

namespace snap {
class LogicalLocation;
class LogicalLocationImpl : public torch::nn::Cloneable<LogicalLocationImpl> {
 public:
  //! data
  int level = 0;
  int64_t lx1 = 0;
  int64_t lx2 = 0;
  int64_t lx3 = 0;

  LogicalLocationImpl() { reset(); }
  LogicalLocationImpl(int level_, int64_t lx1_ = 0, int64_t lx2_ = 0,
                      int64_t lx3_ = 0);
  void reset() override;
  void pretty_print(std::ostream &os) const override;
  std::string to_string() const;

  // comparison functions for sorting
  bool lesser(const LogicalLocation &other) const;
  bool greater(const LogicalLocation &other) const;
  bool equal(const LogicalLocation &other) const;

  // forward function
  std::vector<LogicalLocation> forward() { return {}; }

 protected:
  // These values can exceed the range of int32_t even if the root grid has only
  // a single MeshBlock if >30 levels of AMR are used, since the corresponding
  // max index = 1*2^31 > INT_MAX = 2^31 -1 for most 32-bit signed integer type
  // impelementations
};
TORCH_MODULE(LogicalLocation);

struct OctTreeOptions {
  OctTreeOptions() = default;
  explicit OctTreeOptions(ParameterInput pin);

  ADD_ARG(int, nb1) = 1;
  ADD_ARG(int, nb2) = 1;
  ADD_ARG(int, nb3) = 1;
  ADD_ARG(int, ndim) = 3;

  // ADD_ARG(bool, is_periodic[3]) = {false, false, false};
  // ADD_ARG(bool, amrflag) = false;
};

class OctTreeNode;
class OctTreeNodeImpl : public torch::nn::Cloneable<OctTreeNodeImpl> {
 public:
  //! options with which this `OctTree` was constructed
  OctTreeOptions options;

  // leaf list
  std::vector<torch::nn::AnyModule> leaves;

  //! logical location of the node
  LogicalLocation loc;

  //! constructor
  explicit OctTreeNodeImpl(OctTreeOptions const &options_);
  OctTreeNodeImpl(OctTreeOptions const &options_, LogicalLocation parent_loc,
                  int n);
  void reset() override;

  int nleaf() const { return 1 << options.ndim(); }
  torch::nn::AnyModule leaf(int i, int j, int k) const {
    return leaves[i + (j << 1) + (k << 2)];
  }

  void add_node(LogicalLocation rloc, int &nnew);
  void add_node_without_refine(LogicalLocation rloc);

  void refine(int &nnew) {}
  void derefine(int &ndel);

  void count(int &num);
  torch::optional<OctTreeNode> find_node(LogicalLocation tloc);

  torch::Tensor forward(std::vector<OctTreeNode> *list);
};
TORCH_MODULE(OctTreeNode);

struct NeighborIndex {
  int ox1, ox2, ox3;
};

int octree_root_level(OctTreeOptions const &op);

class OctTreeImpl : public torch::nn::Cloneable<OctTreeImpl> {
 public:
  //! options with which this `OctTree` was constructed
  OctTreeOptions options;

  //! the root node of the tree
  OctTreeNode root = nullptr;

  // Initialize and register root node
  OctTreeImpl() = default;
  explicit OctTreeImpl(OctTreeOptions const &options);
  void reset() override;

  int root_level() const { return octree_root_level(options); }
  //----------------------------------------------------------------------------------------
  //! \brief find a neighboring block, called from the root of the tree
  //!        If it is coarser or same level, return the pointer to that block.
  //!        If it is a finer block, return the pointer to its parent.
  //!        Note that this function must be called on a completed tree only
  torch::optional<OctTreeNode> find_neighbor(LogicalLocation myloc,
                                             NeighborIndex nx,
                                             BoundaryFlag const *bcs,
                                             bool amrflag = false);

  std::vector<OctTreeNode> forward();
};
TORCH_MODULE(OctTree);

}  // namespace snap

#undef ARG
