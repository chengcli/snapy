#pragma once

// torch
#include <torch/torch.h>

namespace snap {

torch::Tensor conservative_refine(torch::Tensor x);
torch::Tensor conservative_coarsen(torch::Tensor x);

}  // namespace snap
