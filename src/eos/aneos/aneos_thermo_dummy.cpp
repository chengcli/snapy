// torch
#include <torch/torch.h>

// snap
#include "aneos_thermo.hpp"

namespace snap {

__attribute__((weak)) ANEOSThermoImpl::ANEOSThermoImpl(
    const std::string& fname) {
  throw std::runtime_error("ANEOSThermoImpl constructor is not implemented");
}

__attribute__((weak)) void ANEOSThermoImpl::reset() {
  throw std::runtime_error("ANEOSThermoImpl::reset is not implemented");
}

__attribute__((weak)) void ANEOSThermoImpl::pretty_print(
    std::ostream& stream) const {
  throw std::runtime_error("ANEOSThermoImpl::pretty_print is not implemented");
}

__attribute__((weak)) std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
ANEOSThermoImpl::compute(std::string ab,
                         std::vector<torch::Tensor> const& args) {
  throw std::runtime_error("ANEOSThermoImpl::compute is not implemented");
}

}  // namespace snap
