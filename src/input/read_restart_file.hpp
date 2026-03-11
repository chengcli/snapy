#pragma once

// C/C++
#include <cstdio>
#include <cstring>
#include <string>

// torch
#include <torch/script.h>
#include <torch/torch.h>

// snap
#include <snap/layout/layout.hpp>

namespace snap {

using Variables = std::map<std::string, torch::Tensor>;

Variables load_restart(std::string const& path, int block_rank = get_rank());

}  // namespace snap
