// torch
#include <torch/csrc/distributed/c10d/ProcessGroupNCCL.hpp>

// snap
#include "distribute_env.hpp"

namespace snap {

void DistributeEnvImpl::_init_nccl() {
  c10d::ProcessGroupNCCL::Options opts;
  opts.isHighPriorityStream = false;

  // Rank -> GPU mapping
  int device_index = options.local_rank() % torch::cuda::device_count();
  torch::Device device(torch::kCUDA, device_index);
  torch::cuda::set_device(device);

  pg = std::make_shared<c10d::ProcessGroupNCCL>(store, options.rank(),
                                                options.world_size(), opts);

  if (options.verbose()) {
    std::cout << "[Rank " << options.rank()
              << ":" << options.local_rank()
              << "] Using NCCL backend on GPU "
              << device_index << "\n";
  }
}

}  // namespace snap
