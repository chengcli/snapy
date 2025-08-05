#pragma once

// torch
#include <ATen/TensorIterator.h>
#include <ATen/native/DispatchStub.h>

namespace at::native {

using flip_zero_fn = int (*)(at::TensorIterator& iter, int dim, int dir);

DECLARE_DISPATCH(flip_zero_fn, flip_zero);

}  // namespace at::native
