// torch
#include <ATen/Dispatch.h>
#include <ATen/TensorIterator.h>
#include <ATen/native/ReduceOpsUtils.h>
#include <ATen/native/cpu/Loops.h>
#include <torch/torch.h>

// snap
#include "bc_dispatch.hpp"
#include "flip_zero_impl.h"

namespace snap {

int flip_zero_cpu(at::TensorIterator& iter, int dim, int dir) {
  int num_flips = 0;
  int grain_size = iter.numel() / at::get_num_threads();

  AT_DISPATCH_INTEGRAL_TYPES(iter.dtype(), "flip_zero_cpu", [&] {
    auto len = at::native::ensure_nonempty_size(iter.output(), dim);
    auto stride = at::native::ensure_nonempty_stride(iter.output(), dim);

    iter.for_each(
        [&](char** data, const int64_t* strides, int64_t n) {
          for (int i = 0; i < n; i++) {
            auto solid = reinterpret_cast<scalar_t*>(data[0] + i * strides[0]);
            auto dp = reinterpret_cast<scalar_t*>(data[1] + i * strides[1]);
            auto fromLen =
                reinterpret_cast<scalar_t*>(data[2] + i * strides[2]);
            auto fromBit =
                reinterpret_cast<scalar_t*>(data[3] + i * strides[3]);
            auto usedFlip =
                reinterpret_cast<scalar_t*>(data[4] + i * strides[4]);

            num_flips += compute_min_flips<scalar_t>(
                solid, len, /*minRun0=*/3, /*minRun1=*/2,
                /*allowBothFlips=*/0, dir, stride, dp, fromLen, fromBit,
                usedFlip);

            reconstruct_solution<scalar_t>(
                solid, len, /*minRun0=*/3, /*minRun1=*/2,
                /*allowBothFlips=*/0, dir, stride, fromLen, fromBit, usedFlip);
          }
        },
        grain_size);
  });

  return num_flips;
}

}  // namespace snap

namespace at::native {

DEFINE_DISPATCH(flip_zero);

REGISTER_ALL_CPU_DISPATCH(flip_zero, &snap::flip_zero_cpu);

}  // namespace at::native
