// torch
#include <ATen/Dispatch.h>
#include <ATen/TensorIterator.h>
#include <ATen/native/ReduceOpsUtils.h>
#include <ATen/native/cpu/Loops.h>
#include <torch/torch.h>

// snap
#include "coord_dispatch.hpp"
#include "coord_utils_impl.h"
#include "cubed_sphere_utils_impl.h"

namespace snap {
void call_cs_interp_LR_cpu(at::TensorIterator& iter, torch::Tensor usrc) {
  int grain_size = iter.numel() / at::get_num_threads();

  AT_DISPATCH_FLOATING_TYPES(iter.dtype(), "call_cs_interp_LR_cpu", [&] {
    int stride1 = at::native::ensure_nonempty_stride(iter.output(), -1);
    int stride2 = at::native::ensure_nonempty_stride(iter.output(), -2);
    int nghost = at::native::ensure_nonempty_size(iter.output(), -3);
    int N = at::native::ensure_nonempty_size(iter.output(), -2);

    iter.for_each(
        [&](char** data, const int64_t* strides, int64_t n) {
          for (int i = 0; i < n; i++) {
            auto out = reinterpret_cast<scalar_t*>(data[0] + i * strides[0]);
            auto inp = reinterpret_cast<scalar_t*>(data[1] + i * strides[1]);

            auto u = usrc.data_ptr<scalar_t>();
            cs_interp_LR<scalar_t>(out, inp, N, nghost, u, stride2, stride1);
          }
        },
        grain_size);
  });
}

void call_cs_interp_BT_cpu(at::TensorIterator& iter, torch::Tensor usrc) {
  int grain_size = iter.numel() / at::get_num_threads();

  AT_DISPATCH_FLOATING_TYPES(iter.dtype(), "call_cs_interp_BT_cpu", [&] {
    int stride1 = at::native::ensure_nonempty_stride(iter.output(), -1);
    int stride2 = at::native::ensure_nonempty_stride(iter.output(), -2);
    int nghost = at::native::ensure_nonempty_size(iter.output(), -2);
    int N = at::native::ensure_nonempty_size(iter.output(), -3);

    iter.for_each(
        [&](char** data, const int64_t* strides, int64_t n) {
          for (int i = 0; i < n; i++) {
            auto out = reinterpret_cast<scalar_t*>(data[0] + i * strides[0]);
            auto inp = reinterpret_cast<scalar_t*>(data[1] + i * strides[1]);

            auto u = usrc.data_ptr<scalar_t>();
            cs_interp_BT<scalar_t>(out, inp, N, nghost, u, stride2, stride1);
          }
        },
        grain_size);
  });
}

void call_coord_vec_lower_cpu(at::TensorIterator& iter) {
  int grain_size = iter.numel() / at::get_num_threads();

  AT_DISPATCH_FLOATING_TYPES(iter.dtype(), "call_coord_vec_lower_cpu", [&] {
    iter.for_each(
        [&](char** data, const int64_t* strides, int64_t n) {
          for (int i = 0; i < n; i++) {
            auto v2 = reinterpret_cast<scalar_t*>(data[0] + i * strides[0]);
            auto v3 = reinterpret_cast<scalar_t*>(data[1] + i * strides[1]);
            auto cth = reinterpret_cast<scalar_t*>(data[2] + i * strides[2]);
            coord_vec_lower_impl(v2, v3, *cth);
          }
        },
        grain_size);
  });
}

void call_coord_vec_lower_mps(at::TensorIterator& iter) {
  auto v2 = iter.output(0).clone();
  auto v3 = iter.output(1).clone();
  auto cth = iter.input(0);

  iter.output(0).copy_(v2 + v3 * cth);
  iter.output(1).copy_(v3 + v2 * cth);
}

void call_coord_vec_raise_cpu(at::TensorIterator& iter) {
  int grain_size = iter.numel() / at::get_num_threads();

  AT_DISPATCH_FLOATING_TYPES(iter.dtype(), "call_coord_vec_raise_cpu", [&] {
    iter.for_each(
        [&](char** data, const int64_t* strides, int64_t n) {
          for (int i = 0; i < n; i++) {
            auto v2 = reinterpret_cast<scalar_t*>(data[0] + i * strides[0]);
            auto v3 = reinterpret_cast<scalar_t*>(data[1] + i * strides[1]);
            auto cth = reinterpret_cast<scalar_t*>(data[2] + i * strides[2]);
            coord_vec_raise_impl(v2, v3, *cth);
          }
        },
        grain_size);
  });
}

void call_coord_vec_raise_mps(at::TensorIterator& iter) {
  auto v2 = iter.output(0).clone();
  auto v3 = iter.output(1).clone();
  auto cth = iter.input(0);
  auto sth2 = 1. - cth * cth;

  iter.output(0).copy_(v2 / sth2 - v3 * cth / sth2);
  iter.output(1).copy_(-v2 * cth / sth2 + v3 / sth2);
}

}  // namespace snap

namespace at::native {

DEFINE_DISPATCH(call_cs_interp_LR);
DEFINE_DISPATCH(call_cs_interp_BT);
DEFINE_DISPATCH(call_coord_vec_lower);
DEFINE_DISPATCH(call_coord_vec_raise);

REGISTER_ALL_CPU_DISPATCH(call_cs_interp_LR, &snap::call_cs_interp_LR_cpu);
REGISTER_ALL_CPU_DISPATCH(call_cs_interp_BT, &snap::call_cs_interp_BT_cpu);
REGISTER_ALL_CPU_DISPATCH(call_coord_vec_lower,
                          &snap::call_coord_vec_lower_cpu);
REGISTER_ALL_CPU_DISPATCH(call_coord_vec_raise,
                          &snap::call_coord_vec_raise_cpu);

REGISTER_MPS_DISPATCH(call_coord_vec_lower, &snap::call_coord_vec_lower_mps);
REGISTER_MPS_DISPATCH(call_coord_vec_raise, &snap::call_coord_vec_raise_mps);

}  // namespace at::native
