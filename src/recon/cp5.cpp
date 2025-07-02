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

void Center5InterpImpl::left(torch::Tensor w, int dim,
                             torch::Tensor const& out) {
  std::vector<int64_t> squash_dim = {0};
  if (w.device().is_cuda()) {
    squash_dim.push_back(dim);
  }

  auto iter = at::TensorIteratorConfig()
                  .resize_outputs(false)
                  .check_all_same_dtype(true)
                  .declare_static_shape(out.sizes(), squash_dim)
                  .add_output(out)
                  .add_input(w)
                  .build();

  at::native::call_poly5(out.device().type(), iter, cm, dim);
}

void Center5InterpImpl::right(torch::Tensor w, int dim,
                              torch::Tensor const& out) {
  std::vector<int64_t> squash_dim = {0};
  if (w.device().is_cuda()) {
    squash_dim.push_back(dim);
  }

  auto iter = at::TensorIteratorConfig()
                  .resize_outputs(false)
                  .check_all_same_dtype(true)
                  .declare_static_shape(out.sizes(), squash_dim)
                  .add_output(out)
                  .add_input(w)
                  .build();

  at::native::call_poly5(out.device().type(), iter, cp, dim);
}
}  // namespace snap
