// torch
#include <ATen/Dispatch.h>
#include <ATen/TensorIterator.h>
#include <ATen/native/ReduceOpsUtils.h>
#include <ATen/native/cpu/Loops.h>
#include <torch/torch.h>

// snapy
#include "utils_dispatch.hpp"

namespace snap {

void bdot_out_cpu(at::Tensor &out, at::Tensor const &inp1,
                  at::Tensor const &inp2, float scale, int dim) {
  out.set_(scale * (inp1 * inp2).sum(dim));
}

}  // namespace snap

namespace at::native {

DEFINE_DISPATCH(bdot_out);
REGISTER_ALL_CPU_DISPATCH(bdot_out, &snap::bdot_out_cpu);
REGISTER_MPS_DISPATCH(bdot_out, &snap::bdot_out_cpu);

}  // namespace at::native
