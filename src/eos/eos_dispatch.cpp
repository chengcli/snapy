// torch
#include <ATen/Dispatch.h>
#include <ATen/TensorIterator.h>
#include <ATen/native/ReduceOpsUtils.h>
#include <ATen/native/cpu/Loops.h>
#include <torch/torch.h>

// snap
#include "eos_dispatch.hpp"
#include "ideal_gas_impl.h"
// #include "ideal_moist_impl.h"

namespace snap {

void ideal_gas_cons2prim_cpu(at::TensorIterator& iter, float gammad) {
  AT_DISPATCH_FLOATING_TYPES(iter.dtype(), "ideal_gas_cons2prim_cpu", [&] {
    auto stride = at::native::ensure_nonempty_stride(iter.output(), 0);

    iter.for_each([&](char** data, const int64_t* strides, int64_t n) {
      for (int i = 0; i < n; i++) {
        auto prim = reinterpret_cast<scalar_t*>(data[0] + i * strides[0]);
        auto cons = reinterpret_cast<scalar_t*>(data[1] + i * strides[1]);
        auto ke = reinterpret_cast<scalar_t*>(data[2] + i * strides[2]);
        auto ie = reinterpret_cast<scalar_t*>(data[3] + i * strides[3]);
        ideal_gas_cons2prim(prim, cons, ke, ie, gammad, stride);
      }
    });
  });
}

void ideal_gas_cons2prim_mps(at::TensorIterator& iter, float gammad) {
  auto prim = iter.output();
  auto cons = iter.input(0);
  auto ke = iter.input(1);
  auto ie = iter.input(2);

  // den -> den
  prim[Index::IDN] = cons[Index::IDN];

  // mom -> vel
  prim.narrow(0, Index::IVX, 3) =
      cons.narrow(0, Index::IVX, 3) / prim[Index::IDN];

  // pcoord->vec_raise_(prim);

  ke.set_(
      (prim.narrow(0, Index::IVX, 3) * cons.narrow(0, Index::IVX, 3)).sum(0));
  ke *= 0.5;

  torch::sub_out(ie, cons[Index::IPR], ke);

  // eng -> pr
  prim[Index::IPR] = (gammad - 1) * ie;
}

/*void call_ideal_moist_cpu(at::TensorIterator& iter) {
  AT_DISPATCH_FLOATING_TYPES(iter.dtype(), "call_ideal_moist_cpu", [&] {
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
}*/

}  // namespace snap

namespace at::native {

DEFINE_DISPATCH(ideal_gas_cons2prim);

REGISTER_ALL_CPU_DISPATCH(ideal_gas_cons2prim, &snap::ideal_gas_cons2prim_cpu);
REGISTER_MPS_DISPATCH(ideal_gas_cons2prim, &snap::ideal_gas_cons2prim_mps);

}  // namespace at::native
