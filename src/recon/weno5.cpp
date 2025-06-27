// torch
#include <ATen/TensorIterator.h>

// snap
#include <snap/snap.h>

#include "interpolation.hpp"
#include "recon_dispatch.hpp"

namespace snap {

void Weno5InterpImpl::reset() {
  c1m = register_buffer(
      "c1m",
      torch::tensor({-1. / 6., 5. / 6., 1. / 3., 0., 0.}, torch::kFloat64));
  c1p = register_buffer("c1p", c1m.flip({0}));

  c2m = register_buffer(
      "c2m",
      torch::tensor({0., 1. / 3., 5. / 6., -1. / 6., 0.}, torch::kFloat64));
  c2p = register_buffer("c2p", c2m.flip({0}));

  c3m = register_buffer(
      "c3m",
      torch::tensor({0., 0., 11. / 6., -7. / 6., 1. / 3.}, torch::kFloat64));
  c3p = register_buffer("c3p", c3m.flip({0}));

  c4m = register_buffer("c4m",
                        torch::tensor({1., -2., 1., 0., 0.}, torch::kFloat64));
  c4p = register_buffer("c4p", c4m.flip({0}));

  c5m = register_buffer("c5m",
                        torch::tensor({1., -4., 3., 0., 0.}, torch::kFloat64));
  c5p = register_buffer("c5p", c5m.flip({0}));

  c6m = register_buffer("c6m",
                        torch::tensor({0., 1., -2., 1., 0.}, torch::kFloat64));
  c6p = register_buffer("c6p", c6m.flip({0}));

  c7m = register_buffer("c7m",
                        torch::tensor({0., -1., 0., 1., 0.}, torch::kFloat64));
  c7p = register_buffer("c7p", c7m.flip({0}));

  c8m = register_buffer("c8m",
                        torch::tensor({0., 0., 1., -2., 1.}, torch::kFloat64));
  c8p = register_buffer("c8p", c8m.flip({0}));

  c9m = register_buffer("c9m",
                        torch::tensor({0., 0., 3., -4., 1.}, torch::kFloat64));
  c9p = register_buffer("c9p", c9m.flip({0}));
}

torch::Tensor Weno5InterpImpl::forward(torch::Tensor w, int dim) {
  auto vec = w.sizes().vec();
  int nghost = stencils() / 2;

  TORCH_CHECK(w.size(dim) > 2 * nghost, "insufficient width");

  vec[dim] -= 2 * nghost;
  vec.insert(vec.begin(), 2);

  auto result = torch::empty(vec, w.options());

  left(w, dim, result[Index::ILT]);
  right(w, dim, result[Index::IRT]);

  return result;
}

void Weno5InterpImpl::left(torch::Tensor w, int dim, torch::Tensor out) const {
  int len = out.size(dim);

  auto iter = at::TensorIteratorConfig()
                  .add_output(out)
                  .add_owned_const_input(w.narrow(dim, 0, len))
                  .build();

  std::vector<torch::Tensor> args = {w,   c1m, c2m, c3m, c4m,
                                     c5m, c6m, c7m, c8m, c9m};
  at::native::call_weno5(out.device().type(), iter, args, dim, options.scale());
}

void Weno5InterpImpl::right(torch::Tensor w, int dim, torch::Tensor out) const {
  int len = out.size(dim);

  auto iter = at::TensorIteratorConfig()
                  .add_output(out)
                  .add_owned_const_input(w.narrow(dim, 0, len))
                  .build();

  std::vector<torch::Tensor> args = {w,   c1p, c2p, c3p, c4p,
                                     c5p, c6p, c7p, c8p, c9p};
  at::native::call_weno5(out.device().type(), iter, args, dim, options.scale());
}

}  // namespace snap
