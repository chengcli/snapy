// base
#include <configure.h>  // gloo and nccl

// torch
#include <torch/csrc/distributed/c10d/ProcessGroup.hpp>
#include <torch/csrc/distributed/c10d/ProcessGroupGloo.hpp>
#include <torch/csrc/distributed/c10d/TCPStore.hpp>

#include "distributed.hpp"
#include "layout.hpp"

namespace snap {

namespace {

// Global process group - set by Python after
// torch.distributed.init_process_group()
c10::intrusive_ptr<c10d::Backend> g_pg;

}  // anonymous namespace

void set_process_group(c10::intrusive_ptr<c10d::Backend> pg) {
  g_pg = std::move(pg);
}

c10::intrusive_ptr<c10d::Backend> get_process_group() { return g_pg; }

bool is_process_group_initialized() { return g_pg.defined(); }

void destroy_process_group() { g_pg.reset(); }

void init_distributed(LayoutOptions const& options) {
  if (options->verbose()) {
    std::cout << "[Rank " << options->rank() << ":" << options->local_rank()
              << "] Initializing distributed environment\n";
  }

  // 1. Build the store
  c10d::TCPStoreOptions store_op;

  store_op.port = options->master_port();
  store_op.numWorkers = options->world_size();
  store_op.isServer = (options->rank() == options->root_rank());

  auto store =
      at::make_intrusive<c10d::TCPStore>(options->master_addr(), store_op);

  // 2. Create ProcessGroup based on backend
  if (options->backend() == "gloo") {
    _init_distributed_gloo(options, store);
  } else if (options->backend() == "nccl") {
    _init_distributed_nccl(options, store);
  } else {
    throw std::runtime_error("Unsupported BACKEND=" + options->backend());
  }

  snap::get_process_group()->barrier()->wait();

  if (options->verbose()) {
    std::cout << "[Rank " << options->rank() << ":" << options->local_rank()
              << "] Distributed environment initialized with backend="
              << options->backend() << ", world_size=" << options->world_size()
              << "\n";
  }
}

void _init_distributed_gloo(LayoutOptions const& options,
                            c10::intrusive_ptr<c10d::Store> const& store) {
  if (options->verbose()) {
    std::cout << "[Rank " << options->rank() << ":" << options->local_rank()
              << "] Using Gloo backend on CPU\n";
  }

  auto opts = c10d::ProcessGroupGloo::Options::create();
  opts->devices.push_back(c10d::ProcessGroupGloo::createDefaultDevice());

  auto pg = c10::make_intrusive<c10d::ProcessGroupGloo>(
      store, options->rank(), options->world_size(), opts);
  set_process_group(pg);
}

#ifdef NOT_USE_C10D_NCCL
void _init_distributed_nccl(LayoutOptions const& options) {}
#endif

}  // namespace snap
