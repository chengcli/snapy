#pragma once

// torch
#include <ATen/TensorIterator.h>
#include <ATen/native/DispatchStub.h>

namespace at::native {

using vic_stage_fn = void (*)(at::TensorIterator& iter, double dt, double grav,
                              int dir);
using vic_redistribute_fn = void (*)(at::TensorIterator& iter, double dt,
                                     double grav, int dir, int nvapor,
                                     int species_flux);

DECLARE_DISPATCH(vic_stage_fn, vic_assemble_partial);
DECLARE_DISPATCH(vic_stage_fn, vic_assemble_full);
DECLARE_DISPATCH(vic_stage_fn, vic_solve_partial);
DECLARE_DISPATCH(vic_stage_fn, vic_solve_full);
DECLARE_DISPATCH(vic_redistribute_fn, vic_redistribute_partial);
DECLARE_DISPATCH(vic_redistribute_fn, vic_redistribute_full);

}  // namespace at::native
