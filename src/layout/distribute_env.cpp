// snap
#include "distribute_env.hpp"

namespace snap {

DistributeEnvOptions::DistributeEnvOptions() {
  // Override by environment variables if available
  master_addr(get_env("MASTER_ADDR", "127.0.0.1"));
  master_port(std::stoi(get_env("MASTER_PORT", "29500"));
  rank(std::stoi(get_env("WORLD_RANK", "0")));
  world_size(std::stoi(get_env("WORLD_SIZE", "1")));
}

DistributeEnvImpl::DistributeEnvImpl(DistributeEnvOptions const& opts)
    : options(opts) {
  // 1. Build the store
  store = std::make_shared<c10d::TCPStore>(
      options.master_addr(), options.master_port(),
      /*isMaster=*/is_master(),
      /*timeout=*/std::chrono::seconds(300));

  // 2. Create ProcessGroup based on backend
  if (options.backend() == "gloo") {
    _init_gloo();
  } else if (options.backend() == "nccl") {
    _init_nccl();
  } else {
    throw std::runtime_error("Unsupported BACKEND=" + backend);
  }

  if (options.verbose()) {
    std::cout << "[Rank " << rank
              << "] Distributed environment initialized with backend="
              << options.backend() << ", world_size=" << world_size << "\n";
  }
}

void DistributeEnvImpl::_init_gloo() {
  c10d::ProcessGroupGloo::Options opts;
  opts.timeout = std::chrono::seconds(300);

  pg = std::make_shared<c10d::ProcessGroupGloo>(store, options.rank(),
                                                options.world_size(), opts);

  if (options.verbose()) {
    std::cout << "[Rank " << rank << "] Using Gloo backend on CPU\n";
  }
}

void DistributeEnvImpl::_init_nccl() {
  c10d::ProcessGroupNCCL::Options opts;
  opts.isHighPriorityStream = false;

  // Rank -> GPU mapping
  int device_index = rank % torch::cuda::device_count();
  torch::Device device(torch::kCUDA, device_index);
  torch::cuda::set_device(device);

  pg = std::make_shared<c10d::ProcessGroupNCCL>(store, options.rank(),
                                                options.world_size(), opts);

  if (options.verbose()) {
    std::cout << "[Rank " << rank << "] Using NCCL backend on GPU "
              << device_index << "\n";
  }
}

}  // namespace snap
