// base
#include <configure.h>  // nccl

// torch
#include <torch/csrc/distributed/c10d/ProcessGroupNCCL.hpp>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>

// snap
#include <snap/utils/log.hpp>
#include "layout.hpp"

namespace snap {

void LayoutImpl::_init_nccl() {
  // Rank -> GPU mapping
  int device_index;
  if (options->device_id() < 0) {
    device_index = options->local_rank();
    options->device_id(device_index);
  } else {
    device_index = options->device_id();
  }

  TORCH_CHECK(device_index < c10::cuda::device_count(), "[Layout] device_id error");

  torch::Device device(torch::kCUDA, device_index);
  c10::cuda::set_device(device_index);

  TORCH_CHECK(process_group != nullptr,
              "[Layout] process group is not available for nccl backend");
  pg = process_group->getBackend(c10::DeviceType::CUDA);
  TORCH_CHECK(pg != nullptr, "[Layout] nccl backend is unavailable in the "
                             "process group initialized from Python");
  pg->setBoundDeviceId(device);

  if (options->verbose()) {
    std::cout << "[Rank " << options->rank()
              << ":" << options->local_rank()
              << "] Using NCCL backend on GPU "
              << device_index << "\n";
  }
}

void LayoutImpl::_group_start() const {
  if (options->backend() == "nccl") {
    auto* nccl_pg = dynamic_cast<c10d::ProcessGroupNCCL*>(pg.get());
    TORCH_CHECK(nccl_pg != nullptr,
                "[Layout] expected ProcessGroupNCCL for backend=nccl");
    nccl_pg->groupStart();
  }
}

void LayoutImpl::_group_end() const {
  if (options->backend() == "nccl") {
    auto* nccl_pg = dynamic_cast<c10d::ProcessGroupNCCL*>(pg.get());
    TORCH_CHECK(nccl_pg != nullptr,
                "[Layout] expected ProcessGroupNCCL for backend=nccl");
    nccl_pg->groupEnd();
  }
}

void LayoutImpl::_sync_device() const {
  if (options->backend() == "nccl") {
    //at::cuda::getCurrentCUDAStream().synchronize();
    cudaDeviceSynchronize();
  }
}

}  // namespace snap
