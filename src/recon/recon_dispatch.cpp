// torch
#include <ATen/Dispatch.h>
#include <ATen/TensorIterator.h>
#include <ATen/native/ReduceOpsUtils.h>
#include <ATen/native/cpu/Loops.h>
#include <torch/torch.h>

// snap
#include "interp_impl.h"
#include "recon_dispatch.hpp"

namespace snap {

template <int N>
void call_poly_cpu(at::TensorIterator& iter, std::vector<torch::Tensor> payload,
                   int dim) {
  AT_DISPATCH_FLOATING_TYPES(iter.dtype(), "call_poly_cpu", [&] {
    int stride1 = at::native::ensure_nonempty_stride(iter.output(), dim);
    int stride2 = at::native::ensure_nonempty_stride(iter.output(), 0);
    int nvar = at::native::ensure_nonempty_size(iter.output(), 0);

    auto c = payload[1].data_ptr<scalar_t>();

    iter.for_each([&](char** data, const int64_t* strides, int64_t n) {
      for (int i = 0; i < n; i++) {
        auto out = reinterpret_cast<scalar_t*>(data[0] + i * strides[0]);
        auto w = reinterpret_cast<scalar_t*>(data[1] + i * strides[1]);
        interp_poly_impl<scalar_t, N>(out, w, c, stride1, stride2, nvar);
      }
    });
  });
}

template <int N>
void call_poly_mps(at::TensorIterator& iter, std::vector<torch::Tensor> payload,
                   int dim) {
  auto out = iter.output();
  auto w = payload[0];
  auto c = payload[1];
  torch::matmul_out(out, w.unfold(dim, N, 1), c);
}

void call_weno3_cpu(at::TensorIterator& iter,
                    std::vector<torch::Tensor> payload, int dim, bool scale) {
  AT_DISPATCH_FLOATING_TYPES(iter.dtype(), "call_weno3_cpu", [&] {
    int stride1 = at::native::ensure_nonempty_stride(iter.output(), dim);
    int stride2 = at::native::ensure_nonempty_stride(iter.output(), 0);
    int nvar = at::native::ensure_nonempty_size(iter.output(), 0);

    auto c1 = payload[1].data_ptr<scalar_t>();
    auto c2 = payload[2].data_ptr<scalar_t>();
    auto c3 = payload[3].data_ptr<scalar_t>();
    auto c4 = payload[4].data_ptr<scalar_t>();

    iter.for_each([&](char** data, const int64_t* strides, int64_t n) {
      for (int i = 0; i < n; i++) {
        auto out = reinterpret_cast<scalar_t*>(data[0] + i * strides[0]);
        auto w = reinterpret_cast<scalar_t*>(data[1] + i * strides[1]);
        interp_weno3_impl(out, w, c1, c2, c3, c4, stride1, stride2, nvar,
                          scale);
      }
    });
  });
}

void call_weno3_mps(at::TensorIterator& iter,
                    std::vector<torch::Tensor> payload, int dim, bool scale) {
  auto result = iter.output();

  auto w = payload[0];
  auto c1 = payload[1];
  auto c2 = payload[2];
  auto c3 = payload[3];
  auto c4 = payload[4];

  auto wu = w.unfold(dim, 3, 1);
  torch::Tensor wscale;
  if (scale) {
    wscale = wu.abs().mean(-1) + 1.e-10;
    wu /= wscale.unsqueeze(-1);
  }

  auto alpha1 = 1. / 3. / (wu.matmul(c3).square() + 1e-6).square();
  auto alpha2 = 2. / 3. / (wu.matmul(c4).square() + 1e-6).square();

  torch::add_out(result, alpha1 * wu.matmul(c1), alpha2 * wu.matmul(c2));
  result /= alpha1 + alpha2;

  if (scale) {
    result.mul_(wscale);
  }
}

void call_weno5_cpu(at::TensorIterator& iter,
                    std::vector<torch::Tensor> payload, int dim, bool scale) {
  AT_DISPATCH_FLOATING_TYPES(iter.dtype(), "call_weno5_cpu", [&] {
    int stride1 = at::native::ensure_nonempty_stride(iter.output(), dim);
    int stride2 = at::native::ensure_nonempty_stride(iter.output(), 0);
    int nvar = at::native::ensure_nonempty_size(iter.output(), 0);

    auto c1 = payload[1].data_ptr<scalar_t>();
    auto c2 = payload[2].data_ptr<scalar_t>();
    auto c3 = payload[3].data_ptr<scalar_t>();
    auto c4 = payload[4].data_ptr<scalar_t>();
    auto c5 = payload[5].data_ptr<scalar_t>();
    auto c6 = payload[6].data_ptr<scalar_t>();
    auto c7 = payload[7].data_ptr<scalar_t>();
    auto c8 = payload[8].data_ptr<scalar_t>();
    auto c9 = payload[9].data_ptr<scalar_t>();

    iter.for_each([&](char** data, const int64_t* strides, int64_t n) {
      for (int i = 0; i < n; i++) {
        auto out = reinterpret_cast<scalar_t*>(data[0] + i * strides[0]);
        auto w = reinterpret_cast<scalar_t*>(data[1] + i * strides[1]);
        interp_weno5_impl(out, w, c1, c2, c3, c4, c5, c6, c7, c8, c9, stride1,
                          stride2, nvar, scale);
      }
    });
  });
}

void call_weno5_mps(at::TensorIterator& iter,
                    std::vector<torch::Tensor> payload, int dim, bool scale) {
  auto result = iter.output();

  auto w = payload[0];
  auto c1 = payload[1];
  auto c2 = payload[2];
  auto c3 = payload[3];
  auto c4 = payload[4];
  auto c5 = payload[5];
  auto c6 = payload[6];
  auto c7 = payload[7];
  auto c8 = payload[8];
  auto c9 = payload[9];

  auto wu = w.unfold(dim, 5, 1);
  torch::Tensor wscale;
  if (scale) {
    wscale = wu.abs().mean(-1) + 1.e-10;
    wu /= wscale.unsqueeze(-1);
  }

  auto beta1 =
      13. / 12. * wu.matmul(c4).square() + 1. / 4. * wu.matmul(c5).square();
  auto beta2 =
      13. / 12. * wu.matmul(c6).square() + 1. / 4. * wu.matmul(c7).square();
  auto beta3 =
      13. / 12. * wu.matmul(c8).square() + 1. / 4. * wu.matmul(c9).square();

  auto alpha1 = 0.3 / (beta1 + 1e-6).square();
  auto alpha2 = 0.6 / (beta2 + 1e-6).square();
  auto alpha3 = 0.1 / (beta3 + 1e-6).square();

  torch::div_out(
      result,
      alpha1 * wu.matmul(c1) + alpha2 * wu.matmul(c2) + alpha3 * wu.matmul(c3),
      alpha1 + alpha2 + alpha3);

  if (scale) {
    result.mul_(wscale);
  }
}

}  // namespace snap

namespace at::native {

DEFINE_DISPATCH(call_poly3);
DEFINE_DISPATCH(call_poly5);
DEFINE_DISPATCH(call_weno3);
DEFINE_DISPATCH(call_weno5);

REGISTER_ALL_CPU_DISPATCH(call_poly3, &snap::call_poly_cpu<3>);
REGISTER_ALL_CPU_DISPATCH(call_poly5, &snap::call_poly_cpu<5>);
REGISTER_ALL_CPU_DISPATCH(call_weno3, &snap::call_weno3_cpu);
REGISTER_ALL_CPU_DISPATCH(call_weno5, &snap::call_weno5_cpu);

REGISTER_MPS_DISPATCH(call_poly3, &snap::call_poly_mps<3>);
REGISTER_MPS_DISPATCH(call_poly5, &snap::call_poly_mps<5>);
REGISTER_MPS_DISPATCH(call_weno3, &snap::call_weno3_mps);
REGISTER_MPS_DISPATCH(call_weno5, &snap::call_weno5_mps);

}  // namespace at::native
