// base
#include <configure.h>  // nccl

// torch
#include <torch/csrc/distributed/c10d/ProcessGroupNCCL.hpp>
#include <c10/util/intrusive_ptr.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>

// snap
#include <snap/utils/log.hpp>
#include "distributed.hpp"
#include "layout.hpp"

namespace snap {

void LayoutImpl::_group_start() const {
  if (options->backend() == "nccl") {
    auto pg = snap::get_process_group();
    pg->startCoalescing(c10::DeviceType::CUDA);
    /*auto nccl = c10::dynamic_intrusive_pointer_cast<c10d::ProcessGroupNCCL>(
      pg->getBackend(c10::DeviceType::CUDA));
    TORCH_CHECK(nccl, "CUDA tensor must use NCCL backend");*/

    std::cout << "calling group start" << std::endl;
  }
}

c10::intrusive_ptr<c10d::Work> LayoutImpl::_group_end() const {
  if (options->backend() == "nccl") {
    //int dev = options->device_id() >= 0 ? options->device_id()
    //                                    : options->local_rank();
    //c10::cuda::CUDAGuard device_guard{static_cast<c10::DeviceIndex>(dev)};
    //c10::cuda::set_device(dev);

    std::cout << "calling group end" << std::endl;

    auto pg = snap::get_process_group();
    return pg->endCoalescing(c10::DeviceType::CUDA);
    /*auto nccl = c10::dynamic_intrusive_pointer_cast<c10d::ProcessGroupNCCL>(
      pg->getBackend(c10::DeviceType::CUDA));
    TORCH_CHECK(nccl, "CUDA tensor must use NCCL backend");

    torch::Device device(torch::kCUDA, dev);
    c10::cuda::set_device(dev);
    pg->setBoundDeviceId(device);

    nccl->groupEnd();*/
  }

  return nullptr;
}

void LayoutImpl::_sync_device() const {
  if (options->backend() == "nccl") {
    int dev = options->device_id() >= 0 ? options->device_id()
                                        : options->local_rank();
    c10::cuda::CUDAGuard device_guard{static_cast<c10::DeviceIndex>(dev)};
    c10::cuda::set_device(dev);

    at::cuda::getCurrentCUDAStream().synchronize();
    std::cout << "calling sync" << std::endl;
    //cudaDeviceSynchronize();
  }
}

}  // namespace snap
