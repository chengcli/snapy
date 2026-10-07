#pragma once

// torch
#include <ATen/TensorIterator.h>
#include <ATen/native/DispatchStub.h>

namespace at::native {

using vic_stage_fn = void (*)(at::TensorIterator& iter, double dt, double grav,
                              int dir);
using vic_redistribute_fn = void (*)(at::TensorIterator& iter, double dt,
                                     double grav, int dir);

DECLARE_DISPATCH(vic_stage_fn, vic_assemble_partial);
DECLARE_DISPATCH(vic_stage_fn, vic_assemble_full);
DECLARE_DISPATCH(vic_stage_fn, vic_solve_partial);
DECLARE_DISPATCH(vic_stage_fn, vic_solve_full);
DECLARE_DISPATCH(vic_redistribute_fn, vic_redistribute_partial);
DECLARE_DISPATCH(vic_redistribute_fn, vic_redistribute_full);

}  // namespace at::native

namespace snap {

//! vic_assemble_*: added to dir, the energy row books the face gravity work of
//! the linearised x1 mass flux instead of the cell work (cartesian x1, #283)
constexpr int kVicFaceWork = 16;

}  // namespace snap
