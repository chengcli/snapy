#pragma once

// torch
#include <torch/nn/cloneable.h>
#include <torch/nn/module.h>
#include <torch/nn/modules/common.h>

// base
#include <snap/add_arg.h>

// snap
#include <snap/snap.h>

#include <snap/input/parameter_input.hpp>

#include "condensation.hpp"

namespace snap {
struct ThermodynamicsOptions {
  ThermodynamicsOptions() = default;
  explicit ThermodynamicsOptions(ParameterInput pin);

  ADD_ARG(double, Rd) = 287.0;
  ADD_ARG(double, gammad_ref) = 1.4;
  ADD_ARG(double, Tref) = 300.0;

  ADD_ARG(int, nvapor) = 0;
  ADD_ARG(int, ncloud) = 0;
  ADD_ARG(std::vector<std::string>, species);
  ADD_ARG(std::vector<double>, mu_ratio_m1);
  ADD_ARG(std::vector<double>, cv_ratio_m1);
  ADD_ARG(std::vector<double>, cp_ratio_m1);
  ADD_ARG(std::vector<double>, h0);

  ADD_ARG(int, max_iter) = 5;
  ADD_ARG(double, ftol) = 1e-2;
  ADD_ARG(double, rtol) = 1e-4;
  ADD_ARG(double, boost) = 256;
  ADD_ARG(CondensationOptions, cond);
};

class ThermodynamicsImpl : public torch::nn::Cloneable<ThermodynamicsImpl> {
 public:
  //! options with which this `Thermodynamics` was constructed
  ThermodynamicsOptions options;

  //! mud / mu - 1.
  torch::Tensor mu_ratio_m1;

  //! cv/cvd - 1.
  torch::Tensor cv_ratio_m1;

  //! cp/cpd - 1.
  torch::Tensor cp_ratio_m1;

  //! enthalpy offset at T = Tref
  torch::Tensor h0;

  //! submodules
  Condensation pcond = nullptr;

  ThermodynamicsImpl() = default;
  explicit ThermodynamicsImpl(const ThermodynamicsOptions& options_);
  void reset() override;

  int species_index(std::string const& name) const;

  virtual torch::Tensor get_gammad(torch::Tensor var,
                                   int type = kPrimitive) const;

  //! \brief Inverse of the mean molecular weight
  /*!
   *! Eq.16 in Li2019
   *! $ \frac{R}{R_d} = \frac{\mu_d}{\mu}$
   *! \return $1/\mu$
   */
  torch::Tensor f_eps(torch::Tensor w) const;
  torch::Tensor f_sig(torch::Tensor w) const;
  torch::Tensor f_psi(torch::Tensor w) const;

  torch::Tensor get_mu() const;
  torch::Tensor get_cv_ref() const;
  torch::Tensor get_cp_ref() const;

  //! \brief Calculate temperature from primitive variable
  /*!
   *! $T = p/(\rho R) = p/(\rho \frac{R}{R_d} Rd)$
   *! \return $T$
   */
  torch::Tensor get_temp(torch::Tensor w) const;

  //! \brief Calculate potential temperature from primitive variable
  /*!
   *! $ \theta = T (p0/p)^{R_d/c_p}$
   *! \return $\theta$
   */
  torch::Tensor get_theta_ref(torch::Tensor w, double p0) const;

  // torch::Tensor saturation_surplus(torch::Tensor var, int type = kTPMole)
  // const;

  //! \brief Calculate mole fraction from mass fraction
  torch::Tensor get_mole_fraction(torch::Tensor yfrac) const;

  //! \brief Calculate mass fraction from mole fraction
  torch::Tensor get_mass_fraction(torch::Tensor xfrac) const;

  //! \brief Perform saturation adjustment
  torch::Tensor forward(torch::Tensor u);

  //! \brief Calculate the equilibrium state given temperature and pressure
  torch::Tensor equilibrate_tp(torch::Tensor temp, torch::Tensor pres,
                               torch::Tensor yfrac) const;

 protected:
  //! \brief Effective adiabatic index
  /*!
   *! Eq.71 in Li2019
   */
  torch::Tensor _get_chi_ref(torch::Tensor w) const;

  //! \brief Calculate molar concentration fron conserved variables
  torch::Tensor _get_mole_concentration(torch::Tensor u) const;

  //! \brief Calculate dimensionless internal energy from concentrations
  torch::Tensor _get_internal_energy_RT_ref(torch::Tensor temp) const;
};
TORCH_MODULE(Thermodynamics);

}  // namespace snap
