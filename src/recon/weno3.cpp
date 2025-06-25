// torch
#include <ATen/TensorIterator.h>

// base
#include <configure.h>

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
  int len = out.size(dim);
  auto iter = at::TensorIteratorConfig()
                  .add_output(out)
                  .add_owned_const_input(w.narrow(dim, 0, len))
                  .add_owned_const_input(w.narrow(dim, 1, len))
                  .add_owned_const_input(w.narrow(dim, 2, len))
                  .add_input(c1m)
                  .add_input(c2m)
                  .add_input(c3m)
                  .add_input(c4m)
                  .build();

  at::native::call_weno3(out.device().type(), iter, options.scale());
}

void Weno3InterpImpl::right(torch::Tensor w, int dim, torch::Tensor out) const {
  int len = out.size(dim);
  auto iter = at::TensorIteratorConfig()
                  .add_output(out)
                  .add_owned_const_input(w.narrow(dim, 2, len))
                  .add_owned_const_input(w.narrow(dim, 1, len))
                  .add_owned_const_input(w.narrow(dim, 0, len))
                  .add_input(c1p)
                  .add_input(c2p)
                  .add_input(c3p)
                  .add_input(c4p)
                  .build();

  at::native::call_weno3(out.device().type(), iter, options.scale());
}

torch::Tensor Weno3InterpImpl::right_fallback(torch::Tensor w, int dim) const {
  auto wu = w.unfold(dim, stencils(), 1);
  torch::Tensor scale;
  if (options.scale()) {
    scale = wu.abs().mean(-1) + 1.e-10;
    wu /= scale.unsqueeze(-1);
  }

  auto alpha1 = 1. / 3. / (wu.matmul(c3p).square() + 1e-6).square();
  auto alpha2 = 2. / 3. / (wu.matmul(c4p).square() + 1e-6).square();
  auto result =
      (alpha1 * wu.matmul(c1p) + alpha2 * wu.matmul(c2p)) / (alpha1 + alpha2);

  if (options.scale()) {
    return result * scale;
  } else {
    return result;
  }
}

}  // namespace snap
