// torch
#include <ATen/TensorIterator.h>

// snap
#include <snap/snap.h>

#include "interpolation.hpp"
#include "recon_dispatch.hpp"

namespace snap {

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
  auto iter = at::TensorIteratorConfig()
                  .resize_outputs(false)
                  .check_all_same_dtype(true)
                  .declare_static_shape(out.sizes(), /*squash_dim=*/{0})
                  .add_output(out)
                  .add_input(w)
                  .build();

  at::native::call_poly5(out.device().type(), iter, cm, dim);
}

void Center5InterpImpl::right(torch::Tensor w, int dim,
                              torch::Tensor out) const {
  auto iter = at::TensorIteratorConfig()
                  .resize_outputs(false)
                  .check_all_same_dtype(true)
                  .declare_static_shape(out.sizes(), /*squash_dim=*/{0})
                  .add_output(out)
                  .add_input(w)
                  .build();

  at::native::call_poly5(out.device().type(), iter, cp, dim);
}
}  // namespace snap
