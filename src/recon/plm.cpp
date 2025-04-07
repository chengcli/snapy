// C/C++
#include <limits>

// spdlog
#include <configure.h>
#include <spdlog/spdlog.h>

// global
#include <globals.h>

// fvm
#include <fvm/index.h>

#include "interpolation.hpp"
#include "recon_formatter.hpp"

namespace canoe {
torch::Tensor PLMInterpImpl::forward(torch::Tensor w, int dim) {
  torch::NoGradGuard no_grad;

  auto vec = w.sizes().vec();
  int nghost = stencils() / 2;

  TORCH_CHECK(w.size(dim) > 2 * nghost, "insufficient width");

  vec[dim] -= 2 * nghost;
  vec.insert(vec.begin(), 2);

  auto result = torch::empty(vec, w.options());

  auto size = w.size(dim);
  auto dw = w.narrow(dim, 1, size - 1) - w.narrow(dim, 0, size - 1);
  auto dw2 = dw.narrow(dim, 0, size - 2) * dw.narrow(dim, 1, size - 2);
  auto dwm = 2. * dw2 /
             (dw.narrow(dim, 0, size - 2) + dw.narrow(dim, 1, size - 2) +
              std::numeric_limits<float>::min());
  dwm *= (dw2 >= 0).to(torch::kInt);
  // auto dw2i = (dw2 <= 0).to(torch::kInt);
  // dwm = dw2i * torch::zeros_like(dwm) + (1 - dw2i) * dwm;

  result[index::ILT] = w.narrow(dim, 1, size - 2) - 0.5 * dwm;
  result[index::IRT] = w.narrow(dim, 1, size - 2) + 0.5 * dwm;

  return result;
}
}  // namespace canoe
