#pragma once

// snap
#include "equation_of_state.hpp"

namespace snap {

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
   * If "W->A" is needed, it should be preceded immediately by "W->U" or "W->I".
   * if "WA->L" is needed, it should be preceded mmediately by "W->A".
   *
   * Any steps in between these calls may invalidate the cache.
   */
  torch::Tensor compute(std::string ab,
                        std::vector<torch::Tensor> const& args) override;

 private:
  //! cache
  torch::Tensor _prim, _cons, _gamma, _ct, _cs, _ke, _ie, _ce, _rhoc;

  //! \brief Convert primitive variables to conserved variables.
  /*
   * \param[in] prim  primitive variables
   * \param[out] out  conserved variables
   */
  void _prim2cons(torch::Tensor prim, torch::Tensor& out);

  //! \brief calculate internal energy
  /*
   * \param[in] prim  primitive variables
   * \param[out] out  internal energy
   */
  void _prim2intEng(torch::Tensor prim, torch::Tensor& out);

  //! \brief calculate temperature.
  /*
   * \param[in] prim  primitive variables
   * \param[out] out  temperature
   */
  void _prim2temp(torch::Tensor prim, torch::Tensor& out);

  //! \brief calculate cloud energy (internal + kinetic).
  /*
   * \param[in] prim  primitive variables
   * \param[out] out  individual cloud energy
   */
  void _prim2cloudEng(torch::Tensor prim, torch::Tensor& out);

  //! \brief Convert conserved variables to primitive variables.
  /*
   * \param[in] cons  conserved variables
   * \param[ou] out   primitive variables
   */
  void _cons2prim(torch::Tensor cons, torch::Tensor& out);

  //! \brief Convert conserved variables to kinetic energy.
  /*
   * * \param[in] cons    conserved variables
   * * \param[out] out    kinetic energy
   */
  void _cons2ke(torch::Tensor cons, torch::Tensor& out);

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

}  // namespace snap
