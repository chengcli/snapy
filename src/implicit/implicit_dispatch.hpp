#pragma once

// torch
#include <ATen/TensorIterator.h>
#include <ATen/native/DispatchStub.h>

namespace at::native {

using vic_solve_fn = void (*)(at::TensorIterator &iter, double dt, double grav,
                              int dir);

DECLARE_DISPATCH(vic_solve_fn, vic_solve_partial);
DECLARE_DISPATCH(vic_solve_fn, vic_solve_full);

}  // namespace at::native
