#pragma once

// torch
#include <torch/nn/cloneable.h>
#include <torch/nn/module.h>
#include <torch/nn/modules/common.h>
#include <torch/nn/modules/container/modulelist.h>

// snap
#include <snap/bc/bc.hpp>
#include <snap/input/parameter_input.hpp>

#include "mesh_functions.hpp"
#include "meshblock.hpp"
#include "oct_tree.hpp"

// arg
#include <snap/add_arg.h>

namespace snap {
struct MeshOptions {
  MeshOptions() = default;

  //! boundary conditions
  std::array<BoundaryFlag, 6> bflags;

  ADD_ARG(double, x1min) = 0.0;
  ADD_ARG(double, x1max) = 1.0;
  ADD_ARG(double, x2min) = 0.0;
  ADD_ARG(double, x2max) = 1.0;
  ADD_ARG(double, x3min) = 0.0;
  ADD_ARG(double, x3max) = 1.0;

  ADD_ARG(int, ncycle) = 1;

  ADD_ARG(MeshGenerator, meshgen1) = UniformMesh;
  ADD_ARG(MeshGenerator, meshgen2) = UniformMesh;
  ADD_ARG(MeshGenerator, meshgen3) = UniformMesh;

  //! submodule options
  ADD_ARG(MeshBlockOptions, block);
  ADD_ARG(OctTreeOptions, tree);
};

class MeshBlock;
class MeshImpl : public torch::nn::Cloneable<MeshImpl> {
 public:
  //! options with which this `Mesh` was constructed
  MeshOptions options;

  //! data
  double current_time, start_time, tlim;

  //! meshblocks
  std::vector<MeshBlock> blocks;

  //! oct-tree
  OctTree tree = nullptr;

  MeshImpl() = default;
  explicit MeshImpl(MeshOptions const& options_);
  void reset() override;

  void ApplyUserWorkBeforeOutput();

  double max_time_step();
  void load_balance();

  //! Advance the simulation to the desired time.
  void forward(double time, int max_steps);

 protected:
  //! timeout
  int timeout_ = 1.0;
};

TORCH_MODULE(Mesh);
}  // namespace snap

#undef ADD_ARG
