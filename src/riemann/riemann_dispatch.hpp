#pragma once

// torch
#include <ATen/TensorIterator.h>
#include <ATen/native/DispatchStub.h>
#include <torch/torch.h>

namespace at::native {

using rsolver_fn = void (*)(at::TensorIterator& iter, int dim);

DECLARE_DISPATCH(rsolver_fn, call_lmars);
DECLARE_DISPATCH(rsolver_fn, call_hllc);

using roe_fn = void (*)(at::TensorIterator& iter, int dim, bool ideal_moist,
                        int nvapor, double gammad,
                        torch::Tensor const& inv_mu_ratio_m1,
                        torch::Tensor const& cv_ratio_m1,
                        torch::Tensor const& u0);
DECLARE_DISPATCH(roe_fn, call_roe);

using shallow_roe_fn = void (*)(at::TensorIterator& iter, int dim, int dir_yz);
DECLARE_DISPATCH(shallow_roe_fn, call_shallow_roe);

}  // namespace at::native
