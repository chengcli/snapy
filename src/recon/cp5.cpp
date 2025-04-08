// torch
#include <ATen/TensorIterator.h>

// base
#include <configure.h>

// snap
#include <snap/snap.h>

#include "interpolation.hpp"
#include "recon_formatter.hpp"

namespace snap {
void call_cp5_cpu(at::TensorIterator& iter);
__attribute__((weak)) void call_cp5_cuda(at::TensorIterator& iter) {}

void Center5InterpImpl::reset() {
  cm = register_buffer("cm", torch::tensor({-1. / 20., 9. / 20., 47. / 60.,
                                            -13. / 60., 1. / 30.},
                                           torch::kFloat64));
  cp = register_buffer("cp", cm.flip({0}));
}

torch::Tensor Center5InterpImpl::forward(torch::Tensor w, int dim) {
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

void Center5InterpImpl::left(torch::Tensor w, int dim,
                             torch::Tensor out) const {
  int nghost = stencils() / 2;
  int len = w.size(dim) - 2 * nghost;
  auto iter = at::TensorIteratorConfig()
                  .add_output(out)
                  .add_owned_const_input(w.narrow(dim, 0, len))
                  .add_owned_const_input(w.narrow(dim, 1, len))
                  .add_owned_const_input(w.narrow(dim, 2, len))
                  .add_owned_const_input(w.narrow(dim, 3, len))
                  .add_owned_const_input(w.narrow(dim, 4, len))
                  .build();

  if (w.is_cpu()) {
    call_cp5_cpu(iter);
  } else if (w.is_cuda()) {
    call_cp5_cuda(iter);
  } else {
    out.copy_(left_fallback(w, dim));
  }
}

void Center5InterpImpl::right(torch::Tensor w, int dim,
                              torch::Tensor out) const {
  int nghost = stencils() / 2;
  int len = w.size(dim) - 2 * nghost;
  auto iter = at::TensorIteratorConfig()
                  .add_output(out)
                  .add_owned_const_input(w.narrow(dim, 4, len))
                  .add_owned_const_input(w.narrow(dim, 3, len))
                  .add_owned_const_input(w.narrow(dim, 2, len))
                  .add_owned_const_input(w.narrow(dim, 1, len))
                  .add_owned_const_input(w.narrow(dim, 0, len))
                  .build();

  if (w.is_cpu()) {
    call_cp5_cpu(iter);
  } else if (w.is_cuda()) {
    call_cp5_cuda(iter);
  } else {
    out.copy_(right_fallback(w, dim));
  }
}
}  // namespace snap
