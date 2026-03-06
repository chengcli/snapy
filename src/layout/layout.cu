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
    auto nccl = c10::dynamic_intrusive_pointer_cast<c10d::ProcessGroupNCCL>(
      pg->getDefaultBackend());
    if (nccl) nccl->groupStart();
  }
}

void LayoutImpl::_group_end() const {
  if (options->backend() == "nccl") {
    auto pg = snap::get_process_group();
    auto nccl = c10::dynamic_intrusive_pointer_cast<c10d::ProcessGroupNCCL>(
      pg->getDefaultBackend());
    if (nccl) nccl->groupEnd();
  }
}

void LayoutImpl::_sync_device() const {
  if (options->backend() == "nccl") {
    //at::cuda::getCurrentCUDAStream().synchronize();
    cudaDeviceSynchronize();
  }
}

}  // namespace snap
