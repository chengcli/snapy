#pragma once

// torch
#include <torch/nn/cloneable.h>
#include <torch/nn/module.h>
#include <torch/nn/modules/common.h>

// snap
#include <snap/eos/equation_of_state.hpp>

// arg
#include <snap/add_arg.h>

namespace YAML {
class Node;
}  // namespace YAML

namespace snap {
struct RiemannSolverOptions {
  static RiemannSolverOptions from_yaml(YAML::Node const& node);
  RiemannSolverOptions() = default;
  void report(std::ostream& os) const {
    os << "* type = " << type() << "\n"
       << "* dir = " << dir() << "\n";
  }

  ADD_ARG(std::string, type) = "roe";

  // used in shallow water equations
  ADD_ARG(std::string, dir) = "omni";

  //! submodule options
  ADD_ARG(EquationOfStateOptions, eos);
};

class RiemannSolverImpl {
 public:
  //! data
  torch::Tensor elr, clr, glr;

  //! options with which this `RiemannSolver` was constructed
  RiemannSolverOptions options;

  //! Solver the Riemann problem
  virtual torch::Tensor forward(torch::Tensor wl, torch::Tensor wr, int dim,
                                torch::Tensor vel_or_flux);

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
                        torch::Tensor vel_or_flux) override;
};
TORCH_MODULE(UpwindSolver);

class RoeSolverImpl : public torch::nn::Cloneable<RoeSolverImpl>,
                      public RiemannSolverImpl {
 public:
  //! submodules
  EquationOfState peos = nullptr;

  //! Constructor to initialize the layers
  RoeSolverImpl() = default;
  explicit RoeSolverImpl(const RiemannSolverOptions& options_)
      : RiemannSolverImpl(options_) {
    reset();
  }
  void reset() override;

  //! Solver the Riemann problem
  torch::Tensor forward(torch::Tensor wl, torch::Tensor wr, int dim,
                        torch::Tensor out) override;
};
TORCH_MODULE(RoeSolver);

class LmarsSolverImpl : public torch::nn::Cloneable<LmarsSolverImpl>,
                        public RiemannSolverImpl {
 public:
  //! submodules
  EquationOfState peos = nullptr;

  //! Constructor to initialize the layers
  LmarsSolverImpl() = default;
  explicit LmarsSolverImpl(const RiemannSolverOptions& options_)
      : RiemannSolverImpl(options_) {
    reset();
  }
  void reset() override;

  //! Solver the Riemann problem
  torch::Tensor forward(torch::Tensor wl, torch::Tensor wr, int dim,
                        torch::Tensor out) override;
};
TORCH_MODULE(LmarsSolver);

class HLLCSolverImpl : public torch::nn::Cloneable<HLLCSolverImpl>,
                       public RiemannSolverImpl {
 public:
  //! submodules
  EquationOfState peos = nullptr;

  //! Constructor to initialize the layers
  HLLCSolverImpl() = default;
  explicit HLLCSolverImpl(const RiemannSolverOptions& options_)
      : RiemannSolverImpl(options_) {
    reset();
  }
  void reset() override;

  //! Solver the Riemann problem
  torch::Tensor forward(torch::Tensor wl, torch::Tensor wr, int dim,
                        torch::Tensor out) override;
};
TORCH_MODULE(HLLCSolver);

class ShallowRoeSolverImpl : public torch::nn::Cloneable<ShallowRoeSolverImpl>,
                             public RiemannSolverImpl {
 public:
  //! submodules
  EquationOfState peos = nullptr;

  //! Constructor to initialize the layers
  ShallowRoeSolverImpl() = default;
  explicit ShallowRoeSolverImpl(const RiemannSolverOptions& options_)
      : RiemannSolverImpl(options_) {
    reset();
  }
  void reset() override;

  //! Solver the Riemann problem
  torch::Tensor forward(torch::Tensor wl, torch::Tensor wr, int dim,
                        torch::Tensor out) override;
};
TORCH_MODULE(ShallowRoeSolver);
}  // namespace snap

#undef ADD_ARG
