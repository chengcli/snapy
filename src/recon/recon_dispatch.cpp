// torch
#include <ATen/Dispatch.h>
#include <ATen/TensorIterator.h>
#include <ATen/native/cpu/Loops.h>
#include <torch/torch.h>

// snap
#include "interp_cp3_impl.hpp"
#include "interp_simple.hpp"
#include "recon_dispatch.hpp"

namespace snap {

void call_cp3_cpu(at::TensorIterator& iter, int dim) {
  AT_DISPATCH_FLOATING_TYPES(iter.dtype(), "call_cp3_cpu", [&] {
    auto stride = at::native::ensure_nonempty_stride(iter.output(), 0);
    iter.for_each([&](char** data, const int64_t* strides, int64_t n) {
      for (int i = 0; i < n; i++) {
        auto out = reinterpret_cast<scalar_t*>(data[0] + i * strides[0]);
        auto w = reinterpret_cast<scalar_t*>(data[1] + i * strides[1]);
        auto c = reinterpret_cast<scalar_t*>(data[2] + i * strides[2]);
        interp_impl<3>(out, w, stride);
      }
    });
  });
}

void call_cp3_mps(at::TensorIterator& iter, int dim) {
  auto out = iter.output();
  auto w = iter.input(0);
  auto c = iter.input(1);
  torch::matmul_out(out, w.unfold(dim, 3, 1), c);
}

void call_cp5_cpu(at::TensorIterator& iter) {
  AT_DISPATCH_FLOATING_TYPES(iter.dtype(), "call_cp5_cpu", [&] {
    at::native::cpu_kernel(iter,
                           [](scalar_t in1, scalar_t in2, scalar_t in3,
                              scalar_t in4, scalar_t in5) -> scalar_t {
                             return interp_cp5(in1, in2, in3, in4, in5);
                           });
  });
}

void call_cp5_mps(at::TensorIterator& iter, int dim) {
  auto out = iter.output();
  auto w = iter.input(0);
  auto c = iter.input(1);
  torch::matmul_out(out, w.unfold(dim, 5, 1), c);
}

void call_weno3_cpu(at::TensorIterator& iter, torch::Tensor, bool scale) {
  AT_DISPATCH_FLOATING_TYPES(iter.dtype(), "call_weno3_cpu", [&] {
    at::native::cpu_kernel(
        iter, [scale](scalar_t in1, scalar_t in2, scalar_t in3) -> scalar_t {
          auto s =
              scale ? (fabs(in1) + fabs(in2) + fabs(in3)) / 3. + 1.e-10 : 1;
          return s * interp_weno3(in1 / s, in2 / s, in3 / s);
        });
  });
}

void call_weno3_mps(at::TensorIterator& iter, bool scale) {
  auto result = iter.output();
  auto w = iter.input(0);
  auto c1 = iter.input(1);
  auto c2 = iter.input(2);
  auto c3 = iter.input(3);
  auto c4 = iter.input(4);

  auto wu = w.unfold(dim, 5, 1);
  torch::Tensor scale;
  if (options.scale()) {
    scale = wu.abs().mean(-1) + 1.e-10;
    wu /= scale.unsqueeze(-1);
  }

  auto alpha1 = 1. / 3. / (wu.matmul(c3).square() + 1e-6).square();
  auto alpha2 = 2. / 3. / (wu.matmul(c4).square() + 1e-6).square();

  torch::add_out(result, alpha1 * wu.matmul(c1), alpha2 * wu.matmul(c2));
  result /= alpha1 + alpha2;

  if (options.scale()) {
    result.mul_(scale);
  }
}

void call_weno5_cpu(at::TensorIterator& iter, bool scale) {
  AT_DISPATCH_FLOATING_TYPES(iter.dtype(), "call_weno5_cpu", [&] {
    at::native::cpu_kernel(
        iter,
        [scale](scalar_t in1, scalar_t in2, scalar_t in3, scalar_t in4,
                scalar_t in5) -> scalar_t {
          auto s = scale ? (fabs(in1) + fabs(in2) + fabs(in3) + fabs(in4) +
                            fabs(in5)) /
                                   5. +
                               1.e-10
                         : 1;
          return s * interp_weno5(in1 / s, in2 / s, in3 / s, in4 / s, in5 / s);
        });
  });
}

}  // namespace snap

namespace at::native {

DEFINE_DISPATCH(call_cp3);
DEFINE_DISPATCH(call_cp5);
DEFINE_DISPATCH(call_weno3);
DEFINE_DISPATCH(call_weno5);

REGISTER_ALL_CPU_DISPATCH(call_cp3, &snap::call_cp3_cpu);
REGISTER_ALL_CPU_DISPATCH(call_cp5, &snap::call_cp5_cpu);
REGISTER_ALL_CPU_DISPATCH(call_weno3, &snap::call_weno3_cpu);
REGISTER_ALL_CPU_DISPATCH(call_weno5, &snap::call_weno5_cpu);

REGISTER_MPS_DISPATCH(call_cp3, &snap::call_cp3_mps);
REGISTER_MPS_DISPATCH(call_cp5, &snap::call_cp5_mps);
REGISTER_MPS_DISPATCH(call_weno3, &snap::call_weno3_mps);
REGISTER_MPS_DISPATCH(call_weno5, &snap::call_weno5_mps);

}  // namespace at::native
