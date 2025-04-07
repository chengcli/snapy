// spdlog
#include <configure.h>
#include <spdlog/spdlog.h>

// global
#include <globals.h>

// torch
#include <ATen/TensorIterator.h>

// fvm
#include <fvm/index.h>

#include "interpolation.hpp"
#include "recon_formatter.hpp"

namespace canoe {
void call_weno5_cpu(at::TensorIterator& iter, bool scale);
void call_weno5_cuda(at::TensorIterator& iter, bool scale);

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

  LOG_INFO(logger, "{} resets with options: {}", name(), options);
}

torch::Tensor Weno5InterpImpl::forward(torch::Tensor w, int dim) {
  torch::NoGradGuard no_grad;

  auto vec = w.sizes().vec();
  int nghost = stencils() / 2;

  TORCH_CHECK(w.size(dim) > 2 * nghost, "insufficient width");

  vec[dim] -= 2 * nghost;
  vec.insert(vec.begin(), 2);

  auto result = torch::empty(vec, w.options());

  left(w, dim, result[index::ILT]);
  right(w, dim, result[index::IRT]);

  return result;
}

void Weno5InterpImpl::left(torch::Tensor w, int dim, torch::Tensor out) const {
  int len = out.size(dim);
  auto iter = at::TensorIteratorConfig()
                  .add_output(out)
                  .add_owned_const_input(w.narrow(dim, 0, len))
                  .add_owned_const_input(w.narrow(dim, 1, len))
                  .add_owned_const_input(w.narrow(dim, 2, len))
                  .add_owned_const_input(w.narrow(dim, 3, len))
                  .add_owned_const_input(w.narrow(dim, 4, len))
                  .build();

  if (w.is_cpu()) {
    call_weno5_cpu(iter, options.scale());
  } else if (w.is_cuda()) {
    call_weno5_cuda(iter, options.scale());
  } else {
    out.copy_(left_fallback(w, dim));
  }
}

void Weno5InterpImpl::right(torch::Tensor w, int dim, torch::Tensor out) const {
  int len = out.size(dim);
  auto iter = at::TensorIteratorConfig()
                  .add_output(out)
                  .add_owned_const_input(w.narrow(dim, 4, len))
                  .add_owned_const_input(w.narrow(dim, 3, len))
                  .add_owned_const_input(w.narrow(dim, 2, len))
                  .add_owned_const_input(w.narrow(dim, 1, len))
                  .add_owned_const_input(w.narrow(dim, 0, len))
                  .build();

  if (w.is_cpu()) {
    call_weno5_cpu(iter, options.scale());
  } else if (w.is_cuda()) {
    call_weno5_cuda(iter, options.scale());
  } else {
    out.copy_(right_fallback(w, dim));
  }
}

torch::Tensor Weno5InterpImpl::left_fallback(torch::Tensor w, int dim) const {
  auto wu = w.unfold(dim, stencils(), 1);
  torch::Tensor scale;
  if (options.scale()) {
    scale = wu.abs().mean(-1) + 1.e-10;
    wu /= scale.unsqueeze(-1);
  }

  auto beta1 =
      13. / 12. * wu.matmul(c4m).square() + 1. / 4. * wu.matmul(c5m).square();
  auto beta2 =
      13. / 12. * wu.matmul(c6m).square() + 1. / 4. * wu.matmul(c7m).square();
  auto beta3 =
      13. / 12. * wu.matmul(c8m).square() + 1. / 4. * wu.matmul(c9m).square();

  auto alpha1 = 0.3 / (beta1 + 1e-6).square();
  auto alpha2 = 0.6 / (beta2 + 1e-6).square();
  auto alpha3 = 0.1 / (beta3 + 1e-6).square();

  auto result = (alpha1 * wu.matmul(c1m) + alpha2 * wu.matmul(c2m) +
                 alpha3 * wu.matmul(c3m)) /
                (alpha1 + alpha2 + alpha3);

  if (options.scale()) {
    return result * scale;
  } else {
    return result;
  }
}

torch::Tensor Weno5InterpImpl::right_fallback(torch::Tensor w, int dim) const {
  auto wu = w.unfold(dim, stencils(), 1);
  torch::Tensor scale;
  if (options.scale()) {
    scale = wu.abs().mean(-1) + 1.e-10;
    wu /= scale.unsqueeze(-1);
  }

  auto beta1 =
      13. / 12. * wu.matmul(c4p).square() + 1. / 4. * wu.matmul(c5p).square();
  auto beta2 =
      13. / 12. * wu.matmul(c6p).square() + 1. / 4. * wu.matmul(c7p).square();
  auto beta3 =
      13. / 12. * wu.matmul(c8p).square() + 1. / 4. * wu.matmul(c9p).square();

  auto alpha1 = 0.3 / (beta1 + 1e-6).square();
  auto alpha2 = 0.6 / (beta2 + 1e-6).square();
  auto alpha3 = 0.1 / (beta3 + 1e-6).square();

  auto result = (alpha1 * wu.matmul(c1p) + alpha2 * wu.matmul(c2p) +
                 alpha3 * wu.matmul(c3p)) /
                (alpha1 + alpha2 + alpha3);

  if (options.scale()) {
    return result * scale;
  } else {
    return result;
  }
}
}  // namespace canoe
