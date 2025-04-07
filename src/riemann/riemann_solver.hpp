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
#include <fvm/coord/coordinate.hpp>
#include <fvm/eos/equation_of_state.hpp>

namespace snap {
struct RiemannSolverOptions {
  RiemannSolverOptions() = default;
  explicit RiemannSolverOptions(ParameterInput pin);

  ADD_ARG(std::string, type) = "roe";
  ADD_ARG(std::string, dir) = "xy";

  //! submodule options
  ADD_ARG(EquationOfStateOptions, eos);
  ADD_ARG(CoordinateOptions, coord);
};

class RiemannSolverImpl {
 public:
  //! options with which this `RiemannSolver` was constructed
  RiemannSolverOptions options;

  //! Solver the Riemann problem
  virtual torch::Tensor forward(torch::Tensor wl, torch::Tensor wr, int dim,
                                torch::Tensor);

 protected:
  //! Disable constructor
  RiemannSolverImpl() = default;
  explicit RiemannSolverImpl(const RiemannSolverOptions& options_);
};

using RiemannSolver = std::shared_ptr<RiemannSolverImpl>;

class UpwindSolverImpl : public torch::nn::Cloneable<UpwindSolverImpl>,
                         public RiemannSolverImpl {
 public:
  //! Constructor to initialize the layers
  UpwindSolverImpl() = default;
  explicit UpwindSolverImpl(const RiemannSolverOptions& options_)
      : RiemannSolverImpl(options_) {
    reset();
  }
  void reset() override {}

  //! Solver the Riemann problem
  torch::Tensor forward(torch::Tensor wl, torch::Tensor wr, int dim,
                        torch::Tensor vel) override;
};
TORCH_MODULE(UpwindSolver);

class RoeSolverImpl : public torch::nn::Cloneable<RoeSolverImpl>,
                      public RiemannSolverImpl {
 public:
  //! submodules
  EquationOfState peos = nullptr;
  Coordinate pcoord = nullptr;

  //! Constructor to initialize the layers
  RoeSolverImpl() = default;
  explicit RoeSolverImpl(const RiemannSolverOptions& options_)
      : RiemannSolverImpl(options_) {
    reset();
  }
  void reset() override;

  //! Solver the Riemann problem
  torch::Tensor forward(torch::Tensor wl, torch::Tensor wr, int dim,
                        torch::Tensor gammad) override;
};
TORCH_MODULE(RoeSolver);

class LmarsSolverImpl : public torch::nn::Cloneable<LmarsSolverImpl>,
                        public RiemannSolverImpl {
 public:
  //! submodules
  EquationOfState peos = nullptr;
  Coordinate pcoord = nullptr;

  //! Constructor to initialize the layers
  LmarsSolverImpl() = default;
  explicit LmarsSolverImpl(const RiemannSolverOptions& options_)
      : RiemannSolverImpl(options_) {
    reset();
  }
  void reset() override;

  //! Solver the Riemann problem
  torch::Tensor forward(torch::Tensor wl, torch::Tensor wr, int dim,
                        torch::Tensor gammad) override;
  torch::Tensor forward_fallback(torch::Tensor wl, torch::Tensor wr, int dim,
                                 torch::Tensor gammad);
};
TORCH_MODULE(LmarsSolver);

class HLLCSolverImpl : public torch::nn::Cloneable<HLLCSolverImpl>,
                       public RiemannSolverImpl {
 public:
  //! submodules
  EquationOfState peos = nullptr;
  Coordinate pcoord = nullptr;

  //! Constructor to initialize the layers
  HLLCSolverImpl() = default;
  explicit HLLCSolverImpl(const RiemannSolverOptions& options_)
      : RiemannSolverImpl(options_) {
    reset();
  }
  void reset() override;

  //! Solver the Riemann problem
  torch::Tensor forward(torch::Tensor wl, torch::Tensor wr, int dim,
                        torch::Tensor gammad) override;

  torch::Tensor forward_fallback(torch::Tensor wl, torch::Tensor wr, int dim,
                                 torch::Tensor gammad);
};
TORCH_MODULE(HLLCSolver);

class ShallowRoeSolverImpl : public torch::nn::Cloneable<ShallowRoeSolverImpl>,
                             public RiemannSolverImpl {
 public:
  //! submodules
  EquationOfState peos = nullptr;
  Coordinate pcoord = nullptr;

  //! Constructor to initialize the layers
  ShallowRoeSolverImpl() = default;
  explicit ShallowRoeSolverImpl(const RiemannSolverOptions& options_)
      : RiemannSolverImpl(options_) {
    reset();
  }
  void reset() override;

  //! Solver the Riemann problem
  torch::Tensor forward(torch::Tensor wl, torch::Tensor wr, int dim,
                        torch::Tensor gammad) override;
};
TORCH_MODULE(ShallowRoeSolver);
}  // namespace snap
