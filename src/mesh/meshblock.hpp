#pragma once

// torch
#include <torch/nn/cloneable.h>
#include <torch/nn/module.h>
#include <torch/nn/modules/common.h>
#include <torch/nn/modules/container/any.h>

// snap
#include <snap/bc/bc_func.hpp>
#include <snap/hydro/hydro.hpp>
#include <snap/intg/integrator.hpp>
#include <snap/layout/distribute_info.hpp>
#include <snap/scalar/scalar.hpp>

// arg
#include <snap/add_arg.h>

namespace snap {

class OutputOptions;

struct MeshBlockOptions {
  static MeshBlockOptions from_yaml(std::string input_file,
                                    DistributeInfo _dist = DistributeInfo());
  MeshBlockOptions() = default;
  void report(std::ostream& os) const {}

  //! output
  ADD_ARG(std::string, basename) = "";
  ADD_ARG(std::vector<OutputOptions>, outputs);

  //! submodule options
  ADD_ARG(IntegratorOptions, intg);
  ADD_ARG(HydroOptions, hydro);
  ADD_ARG(ScalarOptions, scalar);

  //! boundary functions
  ADD_ARG(std::vector<bcfunc_t>, bfuncs);

  //! distributed meshblock info
  ADD_ARG(DistributeInfo, dist);
};

using Variables = std::map<std::string, torch::Tensor>;
class OutputType;

class MeshBlockImpl : public torch::nn::Cloneable<MeshBlockImpl> {
 public:
  //! options with which this `MeshBlock` was constructed
  MeshBlockOptions options;

  //! user output
  std::function<Variables(Variables const&)> user_output_callback;

  //! outputs
  std::vector<std::shared_ptr<OutputType>> output_types;

  //! current cycle number
  int cycle = 0;

  //! submodules
  Integrator pintg = nullptr;
  Hydro phydro = nullptr;
  Scalar pscalar = nullptr;

  //! Constructor to initialize the layers
  MeshBlockImpl() = default;
  explicit MeshBlockImpl(MeshBlockOptions const& options_);
  void reset() override;

  //! \brief return an index tensor for part of the meshblock
  std::vector<torch::indexing::TensorIndex> part(
      std::tuple<int, int, int> offset, bool exterior = true, int extend_x1 = 0,
      int extend_x2 = 0, int extend_x3 = 0) const;

  Variables& initialize(Variables& vars);

  double max_time_step(Variables const& vars);

  Variables& forward(double dt, int stage, Variables& vars);

  void make_outputs(Variables const& vars, double current_time,
                    bool force_write = false);

  void print_cycle_info(double time, double dt) const;

 private:
  //! stage registers
  torch::Tensor _hydro_u0, _hydro_u1;
  torch::Tensor _scalar_s0, _scalar_s1;
};

TORCH_MODULE(MeshBlock);
}  // namespace snap

#undef ADD_ARG
