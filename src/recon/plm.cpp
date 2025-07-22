// C/C++
#include <limits>

// base
#include <configure.h>

// snap
#include <snap/snap.h>

#include "interpolation.hpp"
#include "recon_formatter.hpp"

namespace snap {
std::pair<torch::Tensor, torch::Tensor> PLMInterpImpl::forward(
    torch::Tensor w, int dim, torch::optional<torch::Tensor> wl,
    torch::optional<torch::Tensor> wr) {
  auto vec = w.sizes().vec();
  vec[dim] -= stencils() - 1;  // reduce size by stencils - 1

  auto wlv = wl.value_or(torch::empty(vec, w.options()));
  auto wrv = wr.value_or(torch::empty(vec, w.options()));

  auto size = w.size(dim);
  auto dw = w.narrow(dim, 1, size - 1) - w.narrow(dim, 0, size - 1);
  auto dw2 = dw.narrow(dim, 0, size - 2) * dw.narrow(dim, 1, size - 2);
  auto dwm = 2. * dw2 /
             (dw.narrow(dim, 0, size - 2) + dw.narrow(dim, 1, size - 2) +
              std::numeric_limits<float>::min());
  dwm *= (dw2 >= 0).to(torch::kInt);
  // auto dw2i = (dw2 <= 0).to(torch::kInt);
  // dwm = dw2i * torch::zeros_like(dwm) + (1 - dw2i) * dwm;

  wlv.copy_(w.narrow(dim, 1, size - 2) - 0.5 * dwm);
  wrv.copy_(w.narrow(dim, 1, size - 2) + 0.5 * dwm);

  return std::make_pair(wlv, wrv);
}
}  // namespace snap
