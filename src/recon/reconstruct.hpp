#pragma once

// torch
#include <torch/nn/cloneable.h>
#include <torch/nn/module.h>
#include <torch/nn/modules/common.h>

// snap
#include "interpolation.hpp"

// arg
#include <snap/add_arg.h>

namespace YAML {
class Node;
}  // namespace YAML

namespace snap {
struct ReconstructOptions {
  static ReconstructOptions from_yaml(const YAML::Node& dyn,
                                      std::string section);
  ReconstructOptions() = default;
  void report(std::ostream& os) const {
    interp().report(os);
    os << "* is_boundary_lower = " << (is_boundary_lower() ? "true" : "false")
       << "\n"
       << "* is_boundary_upper = " << (is_boundary_upper() ? "true" : "false")
       << "\n"
       << "* shock = " << (shock() ? "true" : "false") << "\n"
       << "* density_floor = " << density_floor() << "\n"
       << "* pressure_floor = " << pressure_floor() << "\n"
       << "* limiter = " << (limiter() ? "true" : "false") << "\n";
  }

  //! configure options
  ADD_ARG(bool, is_boundary_lower) = false;
  ADD_ARG(bool, is_boundary_upper) = false;
  ADD_ARG(bool, shock) = true;
  ADD_ARG(double, density_floor) = 1.e-10;
  ADD_ARG(double, pressure_floor) = 1.e-10;
  ADD_ARG(bool, limiter) = false;

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

#undef ADD_ARG
