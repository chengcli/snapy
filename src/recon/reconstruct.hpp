#pragma once

// torch
#include <torch/nn/cloneable.h>
#include <torch/nn/module.h>
#include <torch/nn/modules/common.h>

// base
#include <add_arg.h>
#include <configure.h>

#include <input/parameter_input.hpp>

// fvm
#include "interpolation.hpp"

namespace snap {
struct ReconstructOptions {
  ReconstructOptions() = default;
  explicit ReconstructOptions(ParameterInput pin, std::string section,
                              std::string xorder);

  //! configure options
  ADD_ARG(bool, is_boundary_lower) = false;
  ADD_ARG(bool, is_boundary_upper) = false;
  ADD_ARG(bool, shock) = true;

  //! abstract submodules
  ADD_ARG(InterpOptions, interp);
};

class ReconstructImpl : public torch::nn::Cloneable<ReconstructImpl> {
 public:
  //! options with which this `Reconstruction` was constructed
  ReconstructOptions options;

  //! concrete submodules
  Interp pinterp1 = nullptr;
  Interp pinterp2 = nullptr;

  //! Constructor to initialize the layers
  ReconstructImpl() = default;
  explicit ReconstructImpl(const ReconstructOptions& options_);
  void reset() override;

  //! w -> [wl, wr]
  torch::Tensor forward(torch::Tensor w, int dim);
};

TORCH_MODULE(Reconstruct);
}  // namespace snap
