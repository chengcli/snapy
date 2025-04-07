#pragma once

// torch
#include <torch/nn/cloneable.h>
#include <torch/nn/module.h>
#include <torch/nn/modules/common.h>
#include <torch/nn/modules/container/any.h>

// base
#include <configure.h>
#include <add_arg.h>
#include <input/parameter_input.hpp>

// fvm
#include <fvm/hydro/hydro.hpp>
#include <fvm/scalar/scalar.hpp>
#include <fvm/bc/boundary_condition.hpp>
#include <fvm/intg/integrator.hpp>
#include "oct_tree.hpp"

namespace canoe {
struct MeshBlockOptions {
  MeshBlockOptions() = default;
  explicit MeshBlockOptions(ParameterInput pin);

  ADD_ARG(int, nghost) = 1;

  //! submodule options
  ADD_ARG(IntegratorOptions, intg);
  ADD_ARG(HydroOptions, hydro);
  ADD_ARG(ScalarOptions, scalar);
  ADD_ARG(std::vector<BoundaryFlag>, bflags);
  ADD_ARG(torch::nn::AnyModule, loc);
};

class OctTree;
class MeshOptions;
class MeshBlockImpl : public torch::nn::Cloneable<MeshBlockImpl> {
 public:
  //! options with which this `MeshBlock` was constructed
  MeshBlockOptions options;

  //! prognostic data
  torch::Tensor hydro_u, scalar_u;

  //! boundary functions
  std::vector<bfunc_t> bfuncs;

  //! user output variables
  torch::OrderedDict<std::string, torch::Tensor> user_out_var;

  //! submodules
  Integrator pintg = nullptr;
  LogicalLocation ploc = nullptr;
  Hydro phydro = nullptr;
  Scalar pscalar = nullptr;

  //! Constructor to initialize the layers
  MeshBlockImpl() = default;
  explicit MeshBlockImpl(MeshBlockOptions const& options_);
  explicit MeshBlockImpl(MeshBlockOptions const& options_,
                         LogicalLocation ploc_);
  void reset() override;

  int nc1() const { return options.hydro().coord().nc1(); }
  int nc2() const { return options.hydro().coord().nc2(); }
  int nc3() const { return options.hydro().coord().nc3(); }

  //! \brief return an index tensor for part of the meshblock
  std::vector<torch::indexing::TensorIndex> part(
      std::tuple<int, int, int> offset, bool exterior = true, int extend_x1 = 0,
      int extend_x2 = 0, int extend_x3 = 0) const;

  int gid() const { return gid_; }
  void set_gid(int gid) { gid_ = gid; }

  void set_primitives(torch::Tensor hydro_w,
                      torch::optional<torch::Tensor> scalar_w = torch::nullopt);
  void initialize(MeshOptions const& mesh_options, OctTree const& tree);

  double max_root_time_step(
      int root_level, torch::optional<torch::Tensor> solid = torch::nullopt);

  int forward(double dt, int stage,
              torch::optional<torch::Tensor> solid = torch::nullopt);

 protected:
  //! stage registers
  torch::Tensor hydro_u0_, hydro_u1_;
  torch::Tensor scalar_u0_, scalar_u1_;

  int gid_ = 0;
};

TORCH_MODULE(MeshBlock);
}  // namespace canoe
