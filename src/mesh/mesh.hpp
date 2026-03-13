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
  static std::shared_ptr<MeshOptionsImpl> from_yaml(std::string input_file,
                                                    bool verbose = false);

  void report(std::ostream& os) const {
    os << "-- mesh options --\n";
    os << "* blocks_per_process = " << blocks_per_process() << "\n";
    os << "* block = " << (block() != nullptr ? "set" : "null") << "\n";
  }

  ADD_ARG(MeshBlockOptions, block) = nullptr;
  ADD_ARG(int, blocks_per_process) = 1;
};
using MeshOptions = std::shared_ptr<MeshOptionsImpl>;

class MeshImpl : public torch::nn::Cloneable<MeshImpl> {
 public:
  static std::shared_ptr<MeshImpl> from_yaml(std::string input_file,
                                             bool verbose = false);

  MeshOptions options;
  std::vector<MeshBlock> blocks;

  MeshImpl() : options(MeshOptionsImpl::create()) {}
  explicit MeshImpl(MeshOptions const& options_);
  void reset() override;

  double initialize(MeshVariables& vars,
                    std::vector<char const*> const& restart_files = {});
  torch::Device device() const;
  double max_time_step(MeshVariables const& vars);
  void forward(MeshVariables& vars, double dt, int stage);
  void make_outputs(MeshVariables const& vars, double current_time,
                    bool final_write = false);
  void print_cycle_info(MeshVariables const& vars, double time, double dt) const;
  int check_redo(MeshVariables& vars);
  void set_cycle(int cycle);
  void finalize(MeshVariables const& vars, double time);

 private:
};

TORCH_MODULE(Mesh);

}  // namespace snap

#undef ADD_ARG
