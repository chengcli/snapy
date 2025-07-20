#pragma once

// torch
#include <torch/nn/cloneable.h>
#include <torch/nn/module.h>
#include <torch/nn/modules/common.h>

// snap
#include <snap/eos/equation_of_state.hpp>

// arg
#include <snap/add_arg.h>

namespace snap {

struct ImplicitOptions {
  static ImplicitOptions from_yaml(const YAML::Node& root);
  ImplicitOptions() = default;
  void report(std::ostream& os) const {
    os << "* type = " << type() << "\n"
       << "* grav = " << grav() << "\n"
       << "* scheme = " << scheme() << "\n";
  }

  int size() const {
    if (options.scheme() == 1) {  // partial
      return 3;
    } else if (options.scheme() == 9) {  // full
      return 5;
    } else {
      TORCH_CHECK(false, "Unsupported scheme");
    }
  }

  ADD_ARG(std::string, type) = "vic";
  ADD_ARG(double, grav) = 0.;
  ADD_ARG(int, scheme) = 0;

  //! submodules options
  ADD_ARG(EquationOfStateOptions, eos);
};

class ImplicitHydroImpl : public torch::nn::Cloneable<ImplicitHydroImpl> {
 public:
  //! cache
  torch::Tensor wroe, groe, croe;

  //! options with which this `ImplicitHydro` was constructed
  ImplicitOptions options;

  //! submodules
  EquationOfState peos = nullptr;

  //! Constructor to initialize the layer
  ImplicitHydroImpl() = default;
  explicit ImplicitHydroImpl(ImplicitOptions options);
  void reset() override;

  torch::Tensor diffusion_matrix(torch::Tensor wlr, torch::Tensor elr, int dim);

  torch::Tensor flux_jacobian(torch::Tensor w, int dim);

  //! assemble diffusion matrix and flux jacobian
  std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> forward(
      torch::Tensor w, torch::Tensor wlr, torch::Tensor elr, int dim);
};
TORCH_MODULE(ImplicitHydro);

class ImplicitCorrectionImpl
    : public torch::nn::Cloneable<ImplicitCorrectionImpl> {
 public:
  //! options with which this `ImplicitHydro` was constructed
  ImplicitOptions options;

  //! submodules
  ImplicitHydro pvic = nullptr;

  //! Constructor to initialize the layer
  ImplicitCorrectionImpl() = default;
  explicit ImplicitCorrectionImpl(ImplicitOptions options);
  void reset() override;

  torch::Tensor forward(torch::Tensor du, torch::Tensor w, torch::Tensor wlr[3],
                        torch::Tensor elr[3], double dt);
};
TORCH_MODULE(ImplicitCorrection);

}  // namespace snap

#undef ADD_ARG
