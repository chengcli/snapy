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
struct ScalarOptionsImpl {
  static std::shared_ptr<ScalarOptionsImpl> create() {
    return std::make_shared<ScalarOptionsImpl>();
  }
  static std::shared_ptr<ScalarOptionsImpl> from_yaml(
      std::string const& filename, bool verbose = false);
  ScalarOptionsImpl() = default;
  void report(std::ostream& os) const {
    os << "-- scalar options --\n";
    if (thermo()) {
      os << "-- thermo options --\n";
      thermo()->report(os);
    }
    if (kinetics()) {
      os << "-- kinetics options --\n";
      kinetics()->report(os);
    }
  }

  ADD_ARG(bool, verbose) = false;

  //! Thermodynamics options
  ADD_ARG(kintera::ThermoOptions, thermo) = nullptr;

  //! Kinetics options
  ADD_ARG(kintera::KineticsOptions, kinetics) = nullptr;

  //! submodules options
  ADD_ARG(ReconstructOptions, recon) = nullptr;
  ADD_ARG(RiemannSolverOptions, riemann) = nullptr;
};
using ScalarOptions = std::shared_ptr<ScalarOptionsImpl>;

using Variables = std::map<std::string, torch::Tensor>;

class ScalarImpl : public torch::nn::Cloneable<ScalarImpl> {
 public:
  //! \brief Create and register a `Scalar` module
  /*!
   * This function registers the created module as a submodule
   * of the given parent module `p`.
   *
   * \param[in] opts  options for creating the `Scalar` module
   * \param[in] p     parent module for registering the created module
   * \param[in] name  name for registering the created module
   * \return          created `Scalar` module
   */
  static std::shared_ptr<ScalarImpl> create(ScalarOptions const& opts,
                                            torch::nn::Module* p,
                                            std::string const& name = "scalar");

  //! options with which this `Scalar` was constructed
  ScalarOptions options;

  //! submodules
  Coordinate pcoord = nullptr;
  Reconstruct precon = nullptr;
  RiemannSolver priemann = nullptr;

  kintera::ThermoX pthermo = nullptr;
  kintera::Kinetics pkinetics = nullptr;

  //! Constructor to initialize the layers
  ScalarImpl() : options(ScalarOptionsImpl::create()) {}
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
};

TORCH_MODULE(Scalar);
}  // namespace snap
