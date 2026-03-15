// base
#include <configure.h>

// torch
#include <c10/cuda/CUDAFunctions.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/csrc/distributed/c10d/Backend.hpp>
#include <torch/csrc/distributed/c10d/ProcessGroup.hpp>
#include <torch/csrc/distributed/c10d/ProcessGroupNCCL.hpp>
#include <torch/csrc/distributed/c10d/ProcessGroupGloo.hpp>

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
  pg->setBoundDeviceId(device);

  auto backend_nccl =
      c10::static_intrusive_pointer_cast<c10d::Backend>(
          c10::make_intrusive<c10d::ProcessGroupNCCL>(
              store, options_->process_rank(), options_->process_world_size(),
              opts));
  pg->setDefaultBackend(c10d::ProcessGroup::BackendType::NCCL);
  pg->setBackend(c10::DeviceType::CUDA, c10d::ProcessGroup::BackendType::NCCL,
                 backend_nccl);

  if (options_->verbose()) {
    std::cout << "[Process " << options_->process_rank() << ":"
              << options_->local_rank() << "] Using NCCL backend on GPU "
              << device_index << "\n";
  }
}

void ProcessGroupContext::group_start() const {
  if (is_nccl()) {
    pg->startCoalescing(c10::DeviceType::CUDA);
  }
}

void ProcessGroupContext::group_end() const {
  if (is_nccl()) {
    auto work = pg->endCoalescing(c10::DeviceType::CUDA);
    if (work) {
      work->wait();
    }
  }
}

void ProcessGroupContext::sync_stream() const {
  if (is_nccl()) {
    c10::cuda::getCurrentCUDAStream(options_->device_id()).synchronize();
  }
}

void ProcessGroupContext::sync_device() const {
  if (is_nccl()) {
    c10::cuda::device_synchronize();
  }
}

}  // namespace snap
