#pragma once

// torch
#include <torch/torch.h>

namespace snap {

void coord_vec_lower_(torch::Tensor const& vel, torch::Tensor cth);
void coord_vec_raise_(torch::Tensor const& vel, torch::Tensor cth);

}  // namespace snap
