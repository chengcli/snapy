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

  ADD_ARG(std::string, type) = "moist-mixture";
  ADD_ARG(double, density_floor) = 1.e-6;
  ADD_ARG(double, pressure_floor) = 1.e-3;
  ADD_ARG(bool, limiter) = false;

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

 protected:
  //! Disable constructor, to be used only by derived classes.
  EquationOfStateImpl() = default;
  explicit EquationOfStateImpl(EquationOfStateOptions const& options_)
      : options(options_) {}

  //! \brief Apply the conserved variable limiter in place.
  virtual void _apply_conserved_limiter_(torch::Tensor& cons) const;

  //! \brief Apply the primitive variable limiter in place.
  virtual void _apply_primitive_limiter_(torch::Tensor& prim) const;
};

using EquationOfState = std::shared_ptr<EquationOfStateImpl>;

class MoistMixtureImpl final : public torch::nn::Cloneable<MoistMixtureImpl>,
                               public EquationOfStateImpl {
 public:
  //! submodules
  kintera::ThermoY pthermo = nullptr;

  // Constructor to initialize the layers
  MoistMixtureImpl() = default;
  explicit MoistMixtureImpl(EquationOfStateOptions const& options_);
  void reset() override;
  // void pretty_print(std::ostream& os) const override;
  using EquationOfStateImpl::forward;

  int64_t nvar() const override {
    return 4 + pthermo->options.vapor_ids().size() +
           pthermo->options.cloud_ids().size();
  }

  torch::Tensor get_buffer(std::string var) const override {
    return named_buffers()[var];
  }

  //! \brief Implementation of moist mixture equation of state.
  /*!
   * Conversions "W->A" and "WA->L" use cached thermodynamic variables for
   * efficiency.
   *
   * To ensure that the cache is up-to-date, the following order of calls should
   * be followed:
   *
   * If "W->A" is needed, it should be preceded immediately by "W->U".
   * if "WA->L" is needed, it should be preceded mmediately by "W->A".
   *
   * Any steps in between these calls may invalidate the cache.
   */
  torch::Tensor compute(std::string ab,
                        std::vector<torch::Tensor> const& args) override;

 private:
  //! cache
  torch::Tensor _prim, _cons, _gamma, _ct, _cs, _ke;

  //! \brief Convert primitive variables to conserved variables.
  /*
   * \param[in] prim  primitive variables
   * \param[out] out  conserved variables
   */
  void _prim2cons(torch::Tensor prim, torch::Tensor& cons);

  //! \brief Convert conserved variables to primitive variables.
  /*
   * \param[in] cons  conserved variables
   * \param[ou] out   primitive variables
   */
  void _cons2prim(torch::Tensor cons, torch::Tensor& prim);

  //! \brief Compute the adiabatic index
  /*
   * \param[in] ivol  inverse specific volume
   * \param[in] temp  temperature
   * \param[out] out  adiabatic index
   */
  void _adiabatic_index(torch::Tensor ivol, torch::Tensor temp,
                        torch::Tensor& out) const;

  //! \brief Compute the isothermal sound speed
  /*
   * \param[in] temp  temperature
   * \param[in] ivol  inverse specific volume
   * \param[in] dens  total density
   * \param[out] out  isothermal sound speed
   */
  void _isothermal_sound_speed(torch::Tensor ivol, torch::Tensor temp,
                               torch::Tensor dens, torch::Tensor& out) const;
};
TORCH_MODULE(MoistMixture);

class ShallowWaterImpl final : public torch::nn::Cloneable<ShallowWaterImpl>,
                               public EquationOfStateImpl {
 public:
  // Constructor to initialize the layers
  ShallowWaterImpl() = default;
  ShallowWaterImpl(EquationOfStateOptions const& options_);
  void reset() override;
  // void pretty_print(std::ostream& os) const override;
  using EquationOfStateImpl::forward;

  int64_t nvar() const override { return 4; }

  torch::Tensor get_buffer(std::string var) const override {
    return named_buffers()[var];
  }

  //! \brief Implementation of shallow water equation of state.
  torch::Tensor compute(std::string ab,
                        std::vector<torch::Tensor> const& args) override;

 private:
  //! cache
  torch::Tensor _prim, _cons, _cs;

  //! \brief Convert primitive variables to conserved variables.
  /*
   * \param[in] prim  primitive variables
   * \param[out] out  conserved variables
   */
  void _cons2prim(torch::Tensor cons, torch::Tensor& out);

  //! \brief Convert conserved variables to primitive variables.
  /*
   * \param[in] prim  primitive variables
   * \param[out] out  conserved variables
   */
  void _prim2cons(torch::Tensor prim, torch::Tensor& out);

  //! \brief Compute the gravity wave sound speed
  /*
   * \param[in] prim  primitive variables
   * \param[out] out  sound speed
   */
  void _gravity_wave_speed(torch::Tensor prim, torch::Tensor& out) const;
};
TORCH_MODULE(ShallowWater);

}  // namespace snap

#undef ADD_ARG
