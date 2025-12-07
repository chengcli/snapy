// torch
#include <ATen/Dispatch.h>
#include <ATen/TensorIterator.h>
#include <ATen/native/ReduceOpsUtils.h>
#include <ATen/native/cpu/Loops.h>
#include <torch/torch.h>

// snap
#include "coord_dispatch.hpp"
#include "cubed_sphere_interp_impl.h"

namespace snap {
void call_cs_interp_LR_cpu(at::TensorIterator& iter, torch::Tensor usrc) {
  int grain_size = iter.numel() / at::get_num_threads();

  AT_DISPATCH_FLOATING_TYPES(iter.dtype(), "call_cs_interp_LR_cpu", [&] {
    int stride1 = at::native::ensure_nonempty_stride(iter.output(), -1);
    int stride2 = at::native::ensure_nonempty_stride(iter.output(), -2);
    int nghost = at::native::ensure_nonempty_size(iter.output(), -1);
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
    int nghost = at::native::ensure_nonempty_size(iter.output(), -1);
    int N = at::native::ensure_nonempty_size(iter.output(), -2);

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

}  // namespace snap

namespace at::native {

DEFINE_DISPATCH(call_cs_interp_LR);
DEFINE_DISPATCH(call_cs_interp_BT);

REGISTER_ALL_CPU_DISPATCH(call_cs_interp_LR, &snap::call_cs_interp_LR_cpu);
REGISTER_ALL_CPU_DISPATCH(call_cs_interp_BT, &snap::call_cs_interp_BT_cpu);

}  // namespace at::native
