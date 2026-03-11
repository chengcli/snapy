// base
#include <configure.h>

// torch
#include <c10/cuda/CUDAFunctions.h>
#include <torch/csrc/distributed/c10d/ProcessGroupNCCL.hpp>

#include "layout.hpp"
#include "process_group.hpp"

namespace snap {

void ProcessGroupContext::_init_nccl() {
  auto opts = c10d::ProcessGroupNCCL::Options::create();

  int device_index = options_->device_id();
  if (device_index < 0) {
    device_index = options_->local_rank();
    options_->device_id(device_index);
  }

  TORCH_CHECK(device_index < c10::cuda::device_count(),
              "[ProcessGroup] device_id error");

  torch::Device device(torch::kCUDA, device_index);
  c10::cuda::set_device(device_index);

  pg = std::make_shared<c10d::ProcessGroupNCCL>(
      store, options_->process_rank(), options_->process_world_size(), opts);
  pg->setBoundDeviceId(device);

  if (options_->verbose()) {
    std::cout << "[Process " << options_->process_rank() << ":"
              << options_->local_rank() << "] Using NCCL backend on GPU "
              << device_index << "\n";
  }
}

void ProcessGroupContext::group_start() const {
  if (is_nccl()) {
    std::dynamic_pointer_cast<c10d::ProcessGroupNCCL>(pg)->groupStart();
  }
}

void ProcessGroupContext::group_end() const {
  if (is_nccl()) {
    std::dynamic_pointer_cast<c10d::ProcessGroupNCCL>(pg)->groupEnd();
  }
}

void ProcessGroupContext::sync_device() const {
  if (is_nccl()) {
    cudaDeviceSynchronize();
  }
}

}  // namespace snap
