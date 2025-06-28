#pragma once

// torch
#include <ATen/TensorIterator.h>
#include <ATen/native/DispatchStub.h>

namespace at::native {

using recon_poly_fn = void (*)(at::TensorIterator &iter, at::Tensor payload,
                               int dim);

using recon_weno_fn = void (*)(at::TensorIterator &iter, at::Tensor payload,
                               int dim, bool scale);

DECLARE_DISPATCH(recon_poly_fn, call_poly3);
DECLARE_DISPATCH(recon_poly_fn, call_poly5);
DECLARE_DISPATCH(recon_weno_fn, call_weno3);
DECLARE_DISPATCH(recon_weno_fn, call_weno5);

}  // namespace at::native
