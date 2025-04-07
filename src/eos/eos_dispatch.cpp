// torch
#include <ATen/Dispatch.h>
#include <ATen/TensorIterator.h>
#include <ATen/native/ReduceOpsUtils.h>
#include <ATen/native/cpu/Loops.h>

// snap
#include "ideal_gas_impl.h"
#include "ideal_moist_impl.h"

namespace snap {

void call_ideal_gas_cpu(at::TensorIterator& iter) {
  AT_DISPATCH_FLOATING_TYPES(iter.dtype(), "ideal_gas_cpu", [&] {
    auto stride = at::native::ensure_nonempty_stride(iter.output(), 0);

    iter.for_each([&](char** data, const int64_t* strides, int64_t n) {
      for (int i = 0; i < n; i++) {
        auto prim = reinterpret_cast<scalar_t*>(data[0] + i * strides[0]);
        auto cons = reinterpret_cast<scalar_t*>(data[1] + i * strides[1]);
        auto gammad = reinterpret_cast<scalar_t*>(data[2] + i * strides[2]);
        ideal_gas_cons2prim(prim, cons, gammad, stride);
      }
    });
  });
}

void call_ideal_moist_cpu(at::TensorIterator& iter) {
  AT_DISPATCH_FLOATING_TYPES(iter.dtype(), "ideal_moist_cpu", [&] {
    auto stride = at::native::ensure_nonempty_stride(iter.output(), 0);
    auto nhydro = at::native::ensure_nonempty_size(iter.output(), 0);
    auto nmass = nhydro - 5;

    iter.for_each([&](char** data, const int64_t* strides, int64_t n) {
      for (int i = 0; i < n; i++) {
        auto prim = reinterpret_cast<scalar_t*>(data[0] + i * strides[0]);
        auto cons = reinterpret_cast<scalar_t*>(data[1] + i * strides[1]);
        auto gammad = reinterpret_cast<scalar_t*>(data[2] + i * strides[2]);
        auto feps = reinterpret_cast<scalar_t*>(data[3] + i * strides[3]);
        auto fsig = reinterpret_cast<scalar_t*>(data[4] + i * strides[4]);
        ideal_moist_cons2prim(prim, cons, gammad, feps, fsig, nmass, stride);
      }
    });
  });
}

}  // namespace snap
