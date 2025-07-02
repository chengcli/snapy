// base
#include <configure.h>

// snap
#include "interpolation.hpp"

namespace snap {
std::pair<torch::Tensor, torch::Tensor> PPMInterpImpl::forward(
    torch::Tensor w, int dim, torch::optional<torch::Tensor> wl,
    torch::optional<torch::Tensor> wr) {
  auto vec = w.sizes().vec();
  vec[dim] -= stencils() - 1;  // reduce size by stencils - 1

  throw std::runtime_error("PPM interpolation is not implemented yet.");

  return std::make_pair(w, w);
}
}  // namespace snap
