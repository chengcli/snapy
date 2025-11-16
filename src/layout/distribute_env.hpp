// C/C++
#include <iostream>
#include <memory>
#include <string>

// torch
#include <torch/torch.h>

#include <torch/csrc/distributed/c10d/ProcessGroupGloo.hpp>
#include <torch/csrc/distributed/c10d/ProcessGroupNCCL.hpp>
#include <torch/csrc/distributed/c10d/TCPStore.hpp>

// arg
#include <snap/add_arg.h>

namespace snap {

//! Helper to get environment variable with default
static std::string get_env(const char* name, const std::string& def) {
  const char* v = std::getenv(name);
  return v ? std::string(v) : def;
}

struct DistributeEnvOptions {
  DistributeEnvOptions();

  void report(std::ostream& os) const {
    os << "* backend = " << backend() << "\n";
    os << "* master_addr = " << master_addr() << "\n";
    os << "* rank = " << rank() << "\n";
    os << "* world_size = " << world_size() << "\n";
    os << "* master_port = " << master_port() << "\n";
    os << "* verbose = " << (verbose() ? "true" : "false") << "\n";
  }

  ADD_ARG(std::string, backend) = "gloo";
  ADD_ARG(std::string, master_addr) = "127.0.0.1";
  ADD_ARG(int, rank) = 0;
  ADD_ARG(int, world_size) = 1;
  ADD_ARG(int, master_port) = 29500;
  ADD_ARG(bool, verbose) = false;
};

class DistributeEnvImpl {
 public:
  //! options with which this `DistributeEnv` was constructed
  DistributeEnvOptions options;

  std::shared_ptr<c10d::Store> store;
  std::shared_ptr<c10d::ProcessGroup> pg;

  DistributeEnvImpl() = default;
  explicit DistributeEnvImpl(DistributeEnvOptions const& opts);
  virtual ~DistributeEnvImpl() = default;

  bool is_master() const { return options.rank() == 0; }

 private:
  // --- Backend initializers ---
  void _init_gloo();
  void _init_nccl();
};

using DistributeEnv = std::shared_ptr<DistributeEnvImpl>;

}  // namespace snap

#undef ADD_ARG
