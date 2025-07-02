// torch
#include <ATen/TensorIterator.h>

// snap
#include <snap/snap.h>

#include "interpolation.hpp"
#include "recon_dispatch.hpp"

namespace snap {

void Weno5InterpImpl::reset() {
  cm = register_buffer("cm",
                       torch::tensor({{-1. / 6., 5. / 6., 1. / 3., 0., 0.},
                                      {0., 1. / 3., 5. / 6., -1. / 6., 0.},
                                      {0., 0., 11. / 6., -7. / 6., 1. / 3.},
                                      {1., -2., 1., 0., 0.},
                                      {1., -4., 3., 0., 0.},
                                      {0., 1., -2., 1., 0.},
                                      {0., -1., 0., 1., 0.},
                                      {0., 0., 1., -2., 1.},
                                      {0., 0., 3., -4., 1.}},
                                     torch::kFloat64));

  cp = register_buffer("cp", cm.flip({1}));
}

torch::Tensor Weno5InterpImpl::forward(torch::Tensor w, int dim) {
  auto vec = w.sizes().vec();

  vec[dim] -= stencils() - 1;  // reduce size by stencils - 1
  vec.insert(vec.begin(), 2);

  auto result = torch::empty(vec, w.options());

  left(w, dim, result[Index::ILT]);
  right(w, dim, result[Index::IRT]);

  return result;
}

void Weno5InterpImpl::left(torch::Tensor w, int dim, torch::Tensor out) const {
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

  at::native::call_weno5(out.device().type(), iter, cm, dim, options.scale());
}

void Weno5InterpImpl::right(torch::Tensor w, int dim, torch::Tensor out) const {
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

  at::native::call_weno5(out.device().type(), iter, cp, dim, options.scale());
}

}  // namespace snap
