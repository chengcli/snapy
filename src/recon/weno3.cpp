// torch
#include <ATen/TensorIterator.h>

// snap
#include <snap/snap.h>

#include "interpolation.hpp"
#include "recon_dispatch.hpp"

namespace snap {

void Weno3InterpImpl::reset() {
  c1m = register_buffer("c1m",
                        torch::tensor({1. / 2., 1. / 2., 0.}, torch::kFloat64));
  c1p = register_buffer("c1p", c1m.flip({0}));

  c2m = register_buffer(
      "c2m", torch::tensor({0., 3. / 2., -1. / 2.}, torch::kFloat64));
  c2p = register_buffer("c2p", c2m.flip({0}));

  c3m = register_buffer("c3m", torch::tensor({1., -1., 0.}, torch::kFloat64));
  c3p = register_buffer("c3p", c3m.flip({0}));

  c4m = register_buffer("c4m", torch::tensor({0., 1., -1.}, torch::kFloat64));
  c4p = register_buffer("c4p", c4m.flip({0}));
}

torch::Tensor Weno3InterpImpl::forward(torch::Tensor w, int dim) {
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

void Weno3InterpImpl::left(torch::Tensor w, int dim, torch::Tensor out) const {
  auto iter = at::TensorIteratorConfig()
                  .resize_outputs(false)
                  .check_all_same_dtype(true)
                  .declare_static_shape(out.sizes(), /*squash_dim=*/{0})
                  .add_output(out)
                  .add_input(w)
                  .build();

  at::native::call_weno3(out.device().type(), iter, c1m, dim, options.scale());
}

void Weno3InterpImpl::right(torch::Tensor w, int dim, torch::Tensor out) const {
  auto iter = at::TensorIteratorConfig()
                  .resize_outputs(false)
                  .check_all_same_dtype(true)
                  .declare_static_shape(out.sizes(), /*squash_dim=*/{0})
                  .add_output(out)
                  .add_input(w)
                  .build();

  at::native::call_weno3(out.device().type(), iter, c1p, dim, options.scale());
}

}  // namespace snap
