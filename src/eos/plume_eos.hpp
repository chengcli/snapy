#pragma once

// snap
#include "equation_of_state.hpp"

namespace snap {

class PlumeEOSImpl : public torch::nn::Cloneable<PlumeEOSImpl>,
                     public EquationOfStateImpl {
 public:
  // Constructor to initialize the layers
  PlumeEOSImpl() = default;
  explicit PlumeEOSImpl(EquationOfStateOptions const& options_,
                        torch::nn::Module* p = nullptr)
      : EquationOfStateImpl(options_, p) {
    reset();
  }
  void reset() override;
  using EquationOfStateImpl::forward;

  int nvar() const override { return 4; }

  //! The following transformations are need to implement the EOS
  /*!
   * W->U: convert primitive variables to conserved variables
   * U->W: convert conserved variables to primitive variables
   * W->L: compute sound speed from primitive variables (w)
   * W->A: compute adiabatic index from primitive variables (null op)
   * W->T: compute temperature (null op)
   */
  torch::Tensor compute(std::string ab,
                        std::vector<torch::Tensor> const& args) override;

  /*torch::Tensor get_buffer(std::string var) const override {
    return named_buffers()[var];
  }*/

 private:
  //! \brief Convert primitive variables to conserved variables.
  /*!
   * Primitive variables = [R, W, B, V]
   * Conserved variables = [R^2, R^2 W, R^2 B, R^3 V]
   * \param[in] prim  primitive variables
   * \param[out] out  conserved variables
   */
  void _prim2cons(torch::Tensor prim, torch::Tensor& out);

  //! \brief Convert conserved variables to primitive variables.
  /*!
   * Primitive variables = [R, W, B, V]
   * Conserved variables = [R^2, R^2 W, R^2 B, R^3 V]
   * \param[in] cons  conserved variables
   * \param[ou] out   primitive variables
   */
  void _cons2prim(torch::Tensor cons, torch::Tensor& out);
};
TORCH_MODULE(PlumeEOS);

}  // namespace snap
