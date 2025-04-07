// spdlog
#include <configure.h>
#include <spdlog/spdlog.h>

// snap
#include "interpolation.hpp"

namespace snap {
torch::Tensor PPMInterpImpl::forward(torch::Tensor w, int dim) {
  auto vec = w.sizes().vec();
  vec.insert(vec.begin(), 2);
  vec.pop_back();
  return w.unsqueeze(0).expand(vec);
}
}  // namespace snap
