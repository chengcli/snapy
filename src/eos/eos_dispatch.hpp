#pragma once

// torch
#include <ATen/TensorIterator.h>
#include <ATen/native/DispatchStub.h>

namespace at::native {

using ideal_gas_fn = void (*)(at::TensorIterator &iter, double gammad);
// count: the caller reads the failure count (on CUDA a device-to-host read).
// sink: if defined (0-d int32, CUDA), the failures are added to it instead and
// 0 is returned, with no read; the CPU always returns its count.
using fix_vapor_fn = int (*)(at::TensorIterator &iter, bool count,
                             at::Tensor const &sink);

DECLARE_DISPATCH(ideal_gas_fn, ideal_gas_cons2prim);
DECLARE_DISPATCH(fix_vapor_fn, call_fix_vapor);

}  // namespace at::native
