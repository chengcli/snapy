#pragma once

// torch
#include <torch/nn/cloneable.h>
#include <torch/nn/module.h>
#include <torch/nn/modules/common.h>

// snap
#include <snap/coord/coordinate.hpp>
#include <snap/input/parameter_input.hpp>
#include <snap/recon/reconstruct.hpp>
#include <snap/riemann/riemann_solver.hpp>

// arg
#include <snap/add_arg.h>

namespace snap {
struct ScalarOptions {
  ScalarOptions() = default;
  explicit ScalarOptions(ParameterInput pin);

  //! configure options
  ADD_ARG(int, nscalar) = 0;

  //! submodules options
  ADD_ARG(CoordinateOptions, coord);
  ADD_ARG(ReconstructOptions, recon);
  ADD_ARG(RiemannSolverOptions, riemann);
};

class ScalarImpl : public torch::nn::Cloneable<ScalarImpl> {
 public:
  //! options with which this `Scalar` was constructed
  ScalarOptions options;

  //! submodules
  Coordinate pcoord = nullptr;
  Reconstruct precon = nullptr;
  RiemannSolver priemann = nullptr;

  //! Constructor to initialize the layers
  ScalarImpl() = default;
  explicit ScalarImpl(const ScalarOptions& options_);
  void reset() override;

  int nvar() const { return options.nscalar(); }
  virtual double max_time_step(torch::Tensor w) const { return 1.e9; }

  //! Advance the conserved variables by one time step.
  torch::Tensor forward(torch::Tensor scalar_u, double dt);
};

TORCH_MODULE(Scalar);
}  // namespace snap
