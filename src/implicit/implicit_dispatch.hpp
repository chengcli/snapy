#pragma once

// torch
#include <ATen/TensorIterator.h>
#include <ATen/native/DispatchStub.h>

namespace at::native {

using vic_solve_fn = void (*)(at::TensorIterator &iter, double dt, double grav,
                              int il, int iu, int dir);

DECLARE_DISPATCH(vic_solve_fn, vic_solve3);
// DECLARE_DISPATCH(vic_solve_fn, vic_solve5);

}  // namespace at::native
