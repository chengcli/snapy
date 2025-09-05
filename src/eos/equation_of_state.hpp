#pragma once

// torch
#include <torch/nn/cloneable.h>
#include <torch/nn/module.h>
#include <torch/nn/modules/common.h>

// kintera
#include <kintera/thermo/thermo.hpp>

// snap
#include <snap/coord/coordinate.hpp>

// arg
#include <snap/add_arg.h>

namespace YAML {
class Node;
}  // namespace YAML

namespace snap {

struct EquationOfStateOptions {
  static EquationOfStateOptions from_yaml(YAML::Node const& node);
  EquationOfStateOptions() = default;
  void report(std::ostream& os) const {
    os << "* type = " << type() << "\n"
       << "* density_floor = " << density_floor() << "\n"
       << "* pressure_floor = " << pressure_floor() << "\n"
       << "* limiter = " << (limiter() ? "true" : "false") << "\n";
  }

  ADD_ARG(std::string, type) = "moist-mixture";
  ADD_ARG(double, density_floor) = 1.e-10;
  ADD_ARG(double, pressure_floor) = 1.e-10;
  ADD_ARG(double, temperature_floor) = 20.;
  ADD_ARG(bool, limiter) = false;
  ADD_ARG(std::string, eos_file) = "";

  //! submodules options
  ADD_ARG(kintera::ThermoOptions, thermo);
  ADD_ARG(CoordinateOptions, coord);
};

class EquationOfStateImpl {
 public:
  virtual ~EquationOfStateImpl() = default;

  //! options with which this `EquationOfState` was constructed
  EquationOfStateOptions options;

  //! submodules
  Coordinate pcoord = nullptr;

  virtual int64_t nvar() const { return 5; }

  //! \brief Computes hydrodynamic variables from the given abbreviation
  /*!
   * These four abbreviations should be supported:
   *  - "W->U": convert primitive variables to conserved variables
   *  - "U->W": convert conserved variables to primitive variables
   *  - "W->A": compute adiabatic index from conserved variables
   *  - "WA->L": compute sound speed from primitive variables and adiabatic
   * index
   *
   *
   * \param[in] ab    abbreviation for the computation
   * \param[in] args  arguments for the computation
   * \return computed hydrodynamic variables
   */
  virtual torch::Tensor compute(std::string ab,
                                std::vector<torch::Tensor> const& args = {});

  virtual torch::Tensor get_buffer(std::string) const;

  torch::Tensor forward(torch::Tensor cons,
                        torch::optional<torch::Tensor> out = torch::nullopt);

  //! \brief Apply the conserved variable limiter in place.
  virtual void apply_conserved_limiter_(torch::Tensor const& cons);

  //! \brief Apply the primitive variable limiter in place.
  virtual void apply_primitive_limiter_(torch::Tensor const& prim);

 protected:
  //! Disable constructor, to be used only by derived classes.
  EquationOfStateImpl() = default;
  explicit EquationOfStateImpl(EquationOfStateOptions const& options_)
      : options(options_) {}
};

using EquationOfState = std::shared_ptr<EquationOfStateImpl>;

}  // namespace snap

#undef ADD_ARG
