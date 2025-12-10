#pragma once

// torch
#include <torch/torch.h>

namespace snap {

void sph_cart_to_contra_(torch::Tensor const& vel, torch::Tensor phi,
                         torch::Tensor theta);

void sph_contra_to_cart_(torch::Tensor const& vel, torch::Tensor phi,
                         torch::Tensor theta);

}  // namespace snap
