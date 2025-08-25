#pragma once

// torch
#include <torch/nn/cloneable.h>
#include <torch/nn/module.h>
#include <torch/nn/modules/common.h>

// kintera
#include <kintera/kinetics/kinetics.hpp>
#include <kintera/thermo/thermo.hpp>

// snap
#include <snap/coord/coordinate.hpp>
#include <snap/recon/reconstruct.hpp>
#include <snap/riemann/riemann_solver.hpp>

// arg
#include <snap/add_arg.h>

namespace snap {
struct ScalarOptions {
  static ScalarOptions from_yaml(std::string const& filename);
  ScalarOptions() = default;

  //! Thermodynamics options
  ADD_ARG(kintera::ThermoOptions, thermo);

  //! Kinetics options
  ADD_ARG(kintera::KineticsOptions, kinetics);

  //! submodules options
  ADD_ARG(CoordinateOptions, coord);
  ADD_ARG(ReconstructOptions, recon);
  ADD_ARG(RiemannSolverOptions, riemann);
};

using Variables = std::map<std::string, torch::Tensor>;

class ScalarImpl : public torch::nn::Cloneable<ScalarImpl> {
 public:
  //! options with which this `Scalar` was constructed
  ScalarOptions options;

  //! submodules
  Coordinate pcoord = nullptr;
  Reconstruct precon = nullptr;
  RiemannSolver priemann = nullptr;

  kintera::ThermoX pthermo = nullptr;
  kintera::Kinetics pkinetics = nullptr;

  //! Constructor to initialize the layers
  ScalarImpl() = default;
  explicit ScalarImpl(const ScalarOptions& options_);
  void reset() override;

  int nvar() const { return 0; }
  virtual double max_time_step(torch::Tensor w) const { return 1.e9; }

  torch::Tensor get_buffer(std::string var) const {
    return named_buffers()[var];
  }

  //! Advance the conserved variables by one time step.
  torch::Tensor forward(double dt, torch::Tensor scalar_u,
                        Variables const& other);

 private:
  //! cache
  torch::Tensor _X, _V;
};

TORCH_MODULE(Scalar);
}  // namespace snap
