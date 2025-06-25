#pragma once

// torch
#include <ATen/TensorIterator.h>
#include <ATen/native/DispatchStub.h>

namespace at::native {

using recon_cp_fn = void (*)(at::TensorIterator &iter);
using recon_weno_fn = void (*)(at::TensorIterator &iter, bool scale);

DECLARE_DISPATCH(recon_cp_fn, call_cp3);
DECLARE_DISPATCH(recon_cp_fn, call_cp5);
DECLARE_DISPATCH(recon_weno_fn, call_weno3);
DECLARE_DISPATCH(recon_weno_fn, call_weno5);

}  // namespace at::native
