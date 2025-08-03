#pragma once

// snap
#include "equation_of_state.hpp"

namespace snap {

class IdealGasImpl final : public torch::nn::Cloneable<IdealGasImpl>,
                           public EquationOfStateImpl {
 public:
  //! submodules
  kintera::ThermoY pthermo = nullptr;

  // Constructor to initialize the layers
  IdealGasImpl() = default;
  explicit IdealGasImpl(EquationOfStateOptions const& options_);
  void reset() override;
  // void pretty_print(std::ostream& os) const override;
  using EquationOfStateImpl::forward;

  int64_t nvar() const override { return 5; }

  torch::Tensor get_buffer(std::string var) const override {
    return named_buffers()[var];
  }

  //! \brief Implementation of ideal gasequation of state.
  torch::Tensor compute(std::string ab,
                        std::vector<torch::Tensor> const& args) override;

 private:
  //! \brief Convert primitive variables to conserved variables.
  /*
   * \param[in] prim  primitive variables
   * \param[out] out  conserved variables
   */
  void _prim2cons(torch::Tensor prim, torch::Tensor& out);

  //! \brief Convert conserved variables to primitive variables.
  /*
   * \param[in] cons  conserved variables
   * \param[ou] out   primitive variables
   */
  void _cons2prim(torch::Tensor cons, torch::Tensor& out);

  //! \brief calculate internal energy
  /*
   * \param[in] prim  primitive variables
   * \return          internal energy
   */
  torch::Tensor _prim2intEng(torch::Tensor prim);

  //! \brief Convert temperature to kinetic energy.
  /*
   * \param[in] cons    conserved variables
   * \param[in] temp    temperature
   * \return            internal energy
   */
  torch::Tensor _temp2intEng(torch::Tensor cons, torch::Tensor temp);
};
TORCH_MODULE(IdealGas);

}  // namespace snap
