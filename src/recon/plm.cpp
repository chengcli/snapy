// base
#include <configure.h>

// snap
#include <snap/snap.h>

#include "interpolation.hpp"

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
  auto dwl = dw.narrow(dim, 0, size - 2);
  auto dwr = dw.narrow(dim, 1, size - 2);
  auto dw2 = dwl * dwr;
  // dw2 > 0 keeps dwl + dwr away from zero; an epsilon cannot (0/0 at 1e-38)
  auto dwm =
      torch::where(dw2 > 0, 2. * dw2 / (dwl + dwr), torch::zeros_like(dw2));

  wlv.copy_(w.narrow(dim, 1, size - 2) - 0.5 * dwm);
  wrv.copy_(w.narrow(dim, 1, size - 2) + 0.5 * dwm);

  return std::make_pair(wlv, wrv);
}
}  // namespace snap
