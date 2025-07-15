#pragma once

// snap
#include "equation_of_state.hpp"

namespace snap {

class IdealMoistImpl final : public torch::nn::Cloneable<IdealMoistImpl>,
                             public EquationOfStateImpl {
 public:
  //! data
  torch::Tensor inv_mu_ratio_m1, cv_ratio_m1, u0;

  //! submodules
  kintera::ThermoY pthermo = nullptr;

  // Constructor to initialize the layers
  IdealMoistImpl() = default;
  explicit IdealMoistImpl(EquationOfStateOptions const& options_);
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

  //! \brief Implementation of ideal gasequation of state.
  torch::Tensor compute(std::string ab,
                        std::vector<torch::Tensor> const& args) override;

 private:
  //! cache
  torch::Tensor _prim, _cons, _gamma, _cs, _ke, _ie, _ce, _rhoc;

  //! \brief Convert primitive variables to conserved variables.
  /*
   * \param[in] prim  primitive variables
   * \param[out] out  conserved variables
   */
  void _prim2cons(torch::Tensor prim, torch::Tensor& out);

  //! \brief calculate internal energy
  /*
   * \param[in] prim  primitive variables
   * \param[out] out  total internal energy [J/m^3]
   */
  void _prim2intEng(torch::Tensor prim, torch::Tensor& out);

  //! \brief calculate temperature.
  /*
   * \param[in] prim  primitive variables
   * \param[out] out  temperature
   */
  void _prim2temp(torch::Tensor prim, torch::Tensor& out);

  //! \brief calculate cloud energy (internal + kinetic)
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
   * \param[in] cons    conserved variables
   * \param[out] out    kinetic energy
   */
  void _cons2ke(torch::Tensor cons, torch::Tensor& out);

  //! \brief Inverse of the mean molecular weight
  /*!
   *! Eq.16 in Li2019
   *! $ \frac{R}{R_d} = \frac{\mu_d}{\mu}$
   *! \return $1/\mu$
   */
  torch::Tensor _feps(torch::Tensor const& yfrac) const;

  torch::Tensor _fsig(torch::Tensor const& yfrac) const;
};
TORCH_MODULE(IdealMoist);

}  // namespace snap
