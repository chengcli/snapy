#pragma once

// torch
#include <ATen/native/DispatchStub.h>

namespace at::native {

using bdot_out_fn = void (*)(at::Tensor &out, at::Tensor const &inp1,
                             at::Tensor const &inp2, float scale, int dim);

DECLARE_DISPATCH(bdot_out_fn, bdot_out);

}  // namespace at::native
