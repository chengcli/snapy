#pragma once

// torch
#include <ATen/TensorIterator.h>
#include <ATen/native/DispatchStub.h>

namespace at::native {

using ideal_gas_fn = void (*)(at::TensorIterator &iter, double gammad);

DECLARE_DISPATCH(ideal_gas_fn, ideal_gas_cons2prim);

}  // namespace at::native
