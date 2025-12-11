#pragma once

// torch
#include <ATen/TensorIterator.h>
#include <ATen/native/DispatchStub.h>

namespace at::native {

using cs_interp_fn = void (*)(at::TensorIterator &iter, at::Tensor usrc);
using coord_vec_fn = void (*)(at::TensorIterator &iter);

DECLARE_DISPATCH(cs_interp_fn, call_cs_interp_LR);
DECLARE_DISPATCH(cs_interp_fn, call_cs_interp_BT);
DECLARE_DISPATCH(coord_vec_fn, call_coord_vec_lower);
DECLARE_DISPATCH(coord_vec_fn, call_coord_vec_raise);

}  // namespace at::native
