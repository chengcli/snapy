#pragma once

// torch
#include <torch/nn/cloneable.h>
#include <torch/nn/module.h>
#include <torch/nn/modules/common.h>
#include <torch/nn/modules/container/any.h>

// arg
#include <snap/add_arg.h>

// according to:
// https://gkeyll.readthedocs.io/en/latest/dev/ssp-rk.html

namespace snap {
struct IntegratorWeight {
  ADD_ARG(double, wght0) = 0.0;
  ADD_ARG(double, wght1) = 0.0;
  ADD_ARG(double, wght2) = 0.0;
};

struct IntegratorOptions {
  static IntegratorOptions from_yaml(std::string const& filename);
  ADD_ARG(std::string, type) = "rk3";
  ADD_ARG(double, cfl) = 0.9;
};

class IntegratorImpl : public torch::nn::Cloneable<IntegratorImpl> {
 public:
  //! options with which this `Integrator` was constructed
  IntegratorOptions options;

  //! weights for each stage
  std::vector<IntegratorWeight> stages;

  IntegratorImpl() = default;
  explicit IntegratorImpl(IntegratorOptions const& options);
  void reset() override;

  //! \brief compute the average of the three input tensors
  torch::Tensor forward(int stage, torch::Tensor u0, torch::Tensor u1,
                        torch::Tensor u2);
};

TORCH_MODULE(Integrator);
}  // namespace snap

#undef ADD_ARG
