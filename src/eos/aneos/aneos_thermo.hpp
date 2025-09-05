#pragma once

// torch
#include <torch/nn/cloneable.h>
#include <torch/nn/module.h>
#include <torch/nn/modules/common.h>
#include <torch/nn/modules/container/any.h>

namespace snap {

class ANEOSThermoImpl : public torch::nn::Cloneable<ANEOSThermoImpl> {
 public:
  std::string filename;
  torch::Tensor eos_header;
  torch::Tensor eos_data;

  //! cached properties
  std::map<std::string, torch::Tensor> cache;

  ANEOSThermoImpl() = default;
  explicit ANEOSThermoImpl(const std::string& fname);
  void reset() override;
  void pretty_print(std::ostream& stream) const override;

  std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> compute(
      std::string ab, std::vector<torch::Tensor> const& args);

  // forward function is get_pres
  std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> forward(
      torch::Tensor rho, torch::Tensor intEng) {
    return compute("DU->PTL", {rho, intEng});
  }

 private:
  std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> _cal_TUL(
      torch::Tensor rho, torch::Tensor pres);

  std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> _cal_PTL(
      torch::Tensor rho, torch::Tensor intEng);

  std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> _cal_PUL(
      torch::Tensor rho, torch::Tensor temp);
};
TORCH_MODULE(ANEOSThermo);

}  // namespace snap
