#pragma once

// torch
#include <ATen/TensorIterator.h>
#include <ATen/native/DispatchStub.h>

namespace at::native {

using cons2prim_fn = void (*)(at::TensorIterator &iter);

DECLARE_DISPATCH(cons2prim_fn, call_ideal_gas);
DECLARE_DISPATCH(cons2prim_fn, call_ideal_moist);

}  // namespace at::native
