#pragma once

// torch
#include <torch/nn/cloneable.h>
#include <torch/nn/module.h>
#include <torch/nn/modules/common.h>

// base
#include <configure.h>

// kintera
#include <kintera/thermo/thermo.hpp>

// snap
#include <snap/coord/coordinate.hpp>
#include <snap/input/parameter_input.hpp>

// arg
#include <snap/add_arg.h>

namespace snap {

struct EquationOfStateOptions {
  EquationOfStateOptions() = default;

  explicit EquationOfStateOptions(ParameterInput pin);
  ADD_ARG(std::string, type) = "moist_mixture";
  ADD_ARG(double, density_floor) = 1.e-6;
  ADD_ARG(double, pressure_floor) = 1.e-3;
  ADD_ARG(bool, limiter) = false;

  //! submodules options
  ADD_ARG(kintera::ThermoOptions, thermo);
  ADD_ARG(CoordinateOptions, coord);
};

class MoistMixtureImpl : public torch::nn::Cloneable<MoistMixtureImpl> {
 public:
  //! options with which this `EquationOfState` was constructed
  EquationOfStateOptions options;

  //! submodules
  Coordinate pcoord = nullptr;
  kintera::ThermoY pthermo = nullptr;

  // Constructor to initialize the layers
  MoistMixtureImpl() = default;
  explicit MoistMixtureImpl(EquationOfStateOptions const& options_);
  void reset() override;

  int nhydro() const {
    return 5 + pthermo->options.vapor_ids().size() +
           pthermo->options.cloud_ids().size();
  }

  torch::Tensor const& compute(std::string ab,
                               std::vector<torch::Tensor> const& args);

  torch::Tensor forward(torch::Tensor hydro_u,
                        torch::optional<torch::Tensor> out = torch::nullopt);

 private:
  //! cache
  torch::Tensor _prim, _cons, _gamma, _ct;

  //! \brief Convert primitive variables to conserved variables.
  /*
   * \param[in] prim  primitive variables
   * \param[out] out  conserved variables
   */
  void _prim2cons(torch::Tensor prim, torch::Tensor& cons) const;

  //! \brief Convert conserved variables to primitive variables.
  /*
   * \param[in] cons  conserved variables
   * \param[ou] out   primitive variables
   */
  void _cons2prim(torch::Tensor cons, torch::Tensor& prim) const;

  //! \brief Compute the adiabatic index
  /*
   * \param[in] temp  temperature
   * \param[in] ivol  inverse specific volume
   * \param[out] out  adiabatic index
   */
  void _adiabatic_index(torch::Tensor temp, torch::Tensor ivol,
                        torch::Tensor& out) const;

  //! \brief Compute the isothermal sound speed
  /*
   * \param[in] temp  temperature
   * \param[in] ivol  inverse specific volume
   * \param[out] out  isothermal sound speed
   */
  void _isothermal_sound_speed(torch::Tensor temp, torch::Tensor ivol,
                               torch::Tensor& out) const;

  //! \brief Apply the conserved variable limiter in place.
  void _apply_conserved_limiter_(torch::Tensor& cons) const;

  //! \brief Apply the primitive variable limiter in place.
  void _apply_primitive_limiter_(torch::Tensor& prim) const;
};
TORCH_MODULE(MoistMixture);

class ShallowWaterImpl : public torch::nn::Cloneable<ShallowWaterImpl> {
 public:
  // Constructor to initialize the layers
  ShallowWaterImpl() = default;
  ShallowWaterImpl(EquationOfStateOptions const& options_)
      : EquationOfStateImpl(options_) {
    reset();
  }
  void reset() override;

  int nhydro() const override { return 4; }

  torch::Tensor cons2prim(
      torch::Tensor cons,
      torch::optional<torch::Tensor> out = torch::nullopt) const;

  torch::Tensor prim2cons(
      torch::Tensor prim,
      torch::optional<torch::Tensor> out = torch::nullopt) const;

  torch::Tensor sound_speed(
      torch::Tensor prim,
      torch::optional<torch::Tensor> out = torch::nullopt) const;

  torch::Tensor forward(torch::Tensor hydro_u,
                        torch::optional<torch::Tensor> out = torch::nullopt);

 private:
  void apply_primitive_limiter_inplace(torch::Tensor prim);
};
TORCH_MODULE(ShallowWater);

}  // namespace snap

#undef ADD_ARG
