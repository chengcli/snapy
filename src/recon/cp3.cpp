// torch
#include <ATen/TensorIterator.h>

// snap
#include <snap/snap.h>

#include "interpolation.hpp"
#include "recon_dispatch.hpp"

namespace snap {

void Center3InterpImpl::reset() {
  cm = register_buffer(
      "cm", torch::tensor({-1. / 3., 5. / 6., -1. / 6.}, torch::kFloat64));
  cp = register_buffer("cp", cm.flip({0}));
}

torch::Tensor Center3InterpImpl::forward(torch::Tensor w, int dim) {
  torch::NoGradGuard no_grad;

  auto vec = w.sizes().vec();
  int nghost = stencils() / 2;
  vec[dim] -= 2 * nghost;
  vec.insert(vec.begin(), 2);

  auto result = torch::empty(vec, w.options());
  left(w, dim, result[Index::ILT]);
  right(w, dim, result[Index::IRT]);
  return result;
}

void Center3InterpImpl::left(torch::Tensor w, int dim,
                             torch::Tensor out) const {
  int len = out.size(dim);
  auto iter = at::TensorIteratorConfig()
                  .add_output(out)
                  .add_owned_const_input(w.narrow(dim, 0, len))
                  .add_owned_const_input(w.narrow(dim, 1, len))
                  .add_owned_const_input(w.narrow(dim, 2, len))
                  .add_input(cm)
                  .build();

  at::native::call_cp3(out.device().type(), iter, dim);
}

void Center3InterpImpl::right(torch::Tensor w, int dim,
                              torch::Tensor out) const {
  int len = out.size(dim);
  auto iter = at::TensorIteratorConfig()
                  .add_output(out)
                  .add_owned_const_input(w.narrow(dim, 2, len))
                  .add_owned_const_input(w.narrow(dim, 1, len))
                  .add_owned_const_input(w.narrow(dim, 0, len))
                  .add_input(cp)
                  .build();

  at::native::call_cp3(out.device().type(), iter, dim);
}
}  // namespace snap
