#pragma once

// C/C++
#include <functional>
#include <string>
#include <unordered_map>

// torch
#include <torch/torch.h>

// snap
#include "bc.hpp"

using bcfunc_t =
    std::function<void(torch::Tensor const&, int, snap::BoundaryFuncOptions)>;

inline std::unordered_map<std::string, bcfunc_t>& get_bc_func() {
  static std::unordered_map<std::string, bcfunc_t> bcmap;
  return bcmap;
}

struct BCRegistrar {
  BCRegistrar(const std::string& name, bcfunc_t func) {
    get_bc_func()[name] = func;
  }
};

#define BC_FUNCTION(name, var, dim, op)                            \
  void name(torch::Tensor const&, int, snap::BoundaryFuncOptions); \
  static BCRegistrar bc_##name(#name, name);                       \
  void name(torch::Tensor const& var, int dim, snap::BoundaryFuncOptions op)
