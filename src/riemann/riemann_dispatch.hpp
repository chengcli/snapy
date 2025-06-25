#pragma once

// torch
#include <ATen/TensorIterator.h>
#include <ATen/native/DispatchStub.h>

namespace at::native {

using rsolver_fn = void (*)(at::TensorIterator &iter, int dim);

DECLARE_DISPATCH(rsolver_fn, call_lmars);
DECLARE_DISPATCH(rsolver_fn, call_hllc);
DECLARE_DISPATCH(rsolver_fn, call_roe);

}  // namespace at::native
