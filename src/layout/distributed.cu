// base
#include <configure.h>  // nccl

// torch
#include <torch/csrc/distributed/c10d/TCPStore.hpp>
#include <torch/csrc/distributed/c10d/ProcessGroupNCCL.hpp>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>

// snap
#include "distributed.hpp"
#include "layout.hpp"

namespace snap {

void _init_distributed_nccl(LayoutOptions const& options,
                            c10::intrusive_ptr<c10d::Store> const& store) {
  auto opts = c10d::ProcessGroupNCCL::Options::create();

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

  auto pg = c10::make_intrusive<c10d::ProcessGroupNCCL>(
      store, options->rank(), options->world_size(), opts);
  pg->setBoundDeviceId(device);

  if (options->verbose()) {
    std::cout << "[Rank " << options->rank()
              << ":" << options->local_rank()
              << "] Using NCCL backend on GPU "
              << device_index << "\n";
  }

  set_process_group(pg);
}

}  // namespace snap
