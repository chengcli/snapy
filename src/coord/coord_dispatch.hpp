#pragma once

// torch
#include <ATen/TensorIterator.h>
#include <ATen/native/DispatchStub.h>

namespace at::native {

using cs_interp_fn = void (*)(at::TensorIterator &iter, at::Tensor usrc);

DECLARE_DISPATCH(cs_interp_fn, call_cs_interp_LR);
DECLARE_DISPATCH(cs_interp_fn, call_cs_interp_BT);

}  // namespace at::native
