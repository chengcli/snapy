// base
#include <configure.h>  // gloo

// torch
#include <torch/csrc/distributed/c10d/ProcessGroupGloo.hpp>
#include <torch/csrc/distributed/c10d/TCPStore.hpp>

// snap
#include "distribute_env.hpp"

namespace snap {

DistributeEnvOptions::DistributeEnvOptions() {
  // These enrionment variables will be set by torch.distributed.launch
  // Override by them if they are present
  master_addr(get_env("MASTER_ADDR", "127.0.0.1"));
  master_port(std::stoi(get_env("MASTER_PORT", "29500")));
  rank(std::stoi(get_env("RANK", "0")));
  local_rank(std::stoi(get_env("LOCAL_RANK", "0")));
  world_size(std::stoi(get_env("WORLD_SIZE", "1")));
}

DistributeEnvImpl::DistributeEnvImpl(DistributeEnvOptions const& opts)
    : options(opts) {
  // 1. Build the store
  c10d::TCPStoreOptions store_op;

  store_op.port = options.master_port();
  store_op.isServer = options.rank() == 0;
  store_op.numWorkers = options.world_size();

  store = at::make_intrusive<c10d::TCPStore>(options.master_addr(), store_op);

  // 2. Create ProcessGroup based on backend
  if (options.backend() == "gloo") {
    _init_gloo();
  } else if (options.backend() == "nccl") {
    _init_nccl();
  } else {
    throw std::runtime_error("Unsupported BACKEND=" + options.backend());
  }

  if (options.verbose()) {
    std::cout << "[Rank " << options.rank() << ":" << options.local_rank()
              << "] Distributed environment initialized with backend="
              << options.backend() << ", world_size=" << options.world_size()
              << "\n";
  }
}

void DistributeEnvImpl::_init_gloo() {
  pg = std::make_shared<c10d::ProcessGroupGloo>(store, options.rank(),
                                                options.world_size());

  if (options.verbose()) {
    std::cout << "[Rank " << options.rank() << ":" << options.local_rank()
              << "] Using Gloo backend on CPU\n";
  }
}

__attribute__((weak)) void DistributeEnvImpl::_init_nccl() {}

}  // namespace snap
