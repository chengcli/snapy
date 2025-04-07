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

namespace snap {
void call_cp3_cpu(at::TensorIterator& iter);
void call_cp3_cuda(at::TensorIterator& iter);

void Center3InterpImpl::reset() {
  cm = register_buffer(
      "cm", torch::tensor({-1. / 3., 5. / 6., -1. / 6.}, torch::kFloat64));
  cp = register_buffer("cp", cm.flip({0}));

  LOG_INFO(logger, "{} resets with options: {}", name(), options);
}

torch::Tensor Center3InterpImpl::forward(torch::Tensor w, int dim) {
  torch::NoGradGuard no_grad;

  auto vec = w.sizes().vec();
  int nghost = stencils() / 2;
  vec[dim] -= 2 * nghost;
  vec.insert(vec.begin(), 2);

  auto result = torch::empty(vec, w.options());
  left(w, dim, result[index::ILT]);
  right(w, dim, result[index::IRT]);
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
                  .build();

  if (w.is_cpu()) {
    call_cp3_cpu(iter);
  } else if (w.is_cuda()) {
    call_cp3_cuda(iter);
  } else {
    out.copy_(left_fallback(w, dim));
  }
}

void Center3InterpImpl::right(torch::Tensor w, int dim,
                              torch::Tensor out) const {
  int len = out.size(dim);
  auto iter = at::TensorIteratorConfig()
                  .add_output(out)
                  .add_owned_const_input(w.narrow(dim, 2, len))
                  .add_owned_const_input(w.narrow(dim, 1, len))
                  .add_owned_const_input(w.narrow(dim, 0, len))
                  .build();

  if (w.is_cpu()) {
    call_cp3_cpu(iter);
  } else if (w.is_cuda()) {
    call_cp3_cuda(iter);
  } else {
    out.copy_(right_fallback(w, dim));
  }
}
}  // namespace snap
