#pragma once

// C/C++
#include <cstdio>
#include <cstring>
#include <string>

// torch
#include <torch/script.h>
#include <torch/torch.h>

namespace snap {

using Variables = std::map<std::string, torch::Tensor>;

void load_restart(Variables& vars, std::string const& path);

}  // namespace snap
