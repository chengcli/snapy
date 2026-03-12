#pragma once

// C/C++
#include <vector>

// torch
#include <torch/nn/cloneable.h>

// snap
#include <snap/mesh/meshblock.hpp>

// arg
#include <snap/add_arg.h>

namespace snap {

using MeshVariables = std::vector<Variables>;

struct MeshOptionsImpl {
  static std::shared_ptr<MeshOptionsImpl> create() {
    return std::make_shared<MeshOptionsImpl>();
  }

  ADD_ARG(MeshBlockOptions, block) = nullptr;
  ADD_ARG(int, blocks_per_process) = 1;
};
using MeshOptions = std::shared_ptr<MeshOptionsImpl>;

class MeshImpl : public torch::nn::Cloneable<MeshImpl> {
 public:
  MeshOptions options;
  std::vector<MeshBlock> blocks;

  MeshImpl() : options(MeshOptionsImpl::create()) {}
  explicit MeshImpl(MeshOptions const& options_);
  void reset() override;

  void initialize(MeshVariables& vars,
                  std::vector<char const*> const& restart_files = {});
  double max_time_step(MeshVariables const& vars);
  void forward(MeshVariables& vars, double dt, int stage);
  void exchange(MeshVariables& vars, SyncOptions const& opts,
                char const* var_name);

 private:
  void _exchange_all(MeshVariables& vars, SyncOptions const& opts,
                     char const* var_name);
};

TORCH_MODULE(Mesh);

}  // namespace snap

#undef ADD_ARG
