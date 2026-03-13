// C/C++
#include <map>
#include <sstream>

// base
#include <configure.h>

// torch
#include <torch/csrc/distributed/c10d/ProcessGroupGloo.hpp>
#include <torch/csrc/distributed/c10d/TCPStore.hpp>

// snap
#include <snap/utils/log.hpp>

#include "layout.hpp"
#include "process_group.hpp"

namespace snap {

std::mutex ProcessGroupContext::mutex_;

namespace {
std::string process_group_key(LayoutOptions const& opts) {
  std::ostringstream os;
  os << opts->backend() << "|" << opts->master_addr() << "|"
     << opts->master_port() << "|" << opts->process_rank() << "|"
     << opts->process_world_size() << "|" << opts->local_rank() << "|"
     << opts->device_id();
  return os.str();
}
}  // namespace

std::shared_ptr<ProcessGroupContext> ProcessGroupContext::create(
    LayoutOptions const& opts) {
  static std::map<std::string, std::weak_ptr<ProcessGroupContext>> cache;

  std::lock_guard<std::mutex> lock(mutex_);
  auto key = process_group_key(opts);
  auto it = cache.find(key);
  if (it != cache.end()) {
    if (auto existing = it->second.lock()) {
      return existing;
    }
  }

  auto ctx =
      std::shared_ptr<ProcessGroupContext>(new ProcessGroupContext(opts));
  cache[key] = ctx;
  return ctx;
}

ProcessGroupContext::ProcessGroupContext(LayoutOptions const& opts)
    : options_(opts), backend(opts->backend()) {
  _init();
}

void ProcessGroupContext::_init() {
  if (options_->no_backend()) return;

  if (options_->verbose()) {
    std::cout << "[Process " << options_->process_rank() << ":"
              << options_->local_rank()
              << "] Initializing distributed environment\n";
  }

  c10d::TCPStoreOptions store_opts;
  store_opts.port = options_->master_port();
  store_opts.numWorkers = options_->process_world_size();
  store_opts.isServer = options_->process_rank() == 0;
  store =
      at::make_intrusive<c10d::TCPStore>(options_->master_addr(), store_opts);

  if (backend == "gloo") {
    _init_gloo();
  } else if (backend == "nccl") {
    _init_nccl();
  } else {
    throw std::runtime_error("Unsupported BACKEND=" + backend);
  }

  pg->barrier()->wait();

  if (options_->verbose()) {
    std::cout << "[Process " << options_->process_rank() << ":"
              << options_->local_rank()
              << "] Distributed environment initialized with backend="
              << backend << ", world_size=" << options_->process_world_size()
              << "\n";
  }
}

void ProcessGroupContext::_init_gloo() {
  if (options_->verbose()) {
    std::cout << "[Process " << options_->process_rank() << ":"
              << options_->local_rank() << "] Using Gloo backend on CPU\n";
  }

  auto opts = c10d::ProcessGroupGloo::Options::create();
  opts->devices.push_back(c10d::ProcessGroupGloo::createDefaultDevice());

  pg = std::make_shared<c10d::ProcessGroupGloo>(
      store, options_->process_rank(), options_->process_world_size(), opts);
}

#ifdef NOT_USE_C10D_NCCL
void ProcessGroupContext::_init_nccl() {}
void ProcessGroupContext::group_start() const {}
void ProcessGroupContext::group_end() const {}
void ProcessGroupContext::sync_device() const {}
#endif

}  // namespace snap
