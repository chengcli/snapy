// yaml
#include <yaml-cpp/yaml.h>

// base
#include <configure.h>  // gloo

// torch
#include <torch/csrc/distributed/c10d/ProcessGroupGloo.hpp>
#include <torch/csrc/distributed/c10d/TCPStore.hpp>

// snap
#include <snap/mesh/meshblock.hpp>

#include "layout.hpp"

namespace snap {

LayoutOptionsImpl::LayoutOptionsImpl() {
  // These enrionment variables will be set by torch.distributed.launch
  // Override by them if they are present
  master_addr(get_env("MASTER_ADDR", "127.0.0.1"));
  master_port(std::stoi(get_env("MASTER_PORT", "29500")));
  rank(std::stoi(get_env("RANK", "0")));
  local_rank(std::stoi(get_env("LOCAL_RANK", "0")));
  world_size(std::stoi(get_env("WORLD_SIZE", "1")));
}

LayoutOptions LayoutOptionsImpl::from_yaml(std::string const& filename,
                                           bool verbose) {
  auto op = LayoutOptionsImpl::create();
  auto config = YAML::LoadFile(filename);

  if (!config["distribute"]) return op;

  auto node = config["distribute"];

  op->type() = node["layout"].as<std::string>("slab");
  if (op->type() == "slab") {
    op->px(node["nb3"].as<int>(1));
    op->py(node["nb2"].as<int>(1));
    op->pz(node["nb1"].as<int>(1));

    TORCH_CHECK(
        op->pz() == 1,
        "Slab layout only supports partitioning along x2-x3 directions.");

    op->backend() = node["backend"].as<std::string>("gloo");
    op->verbose() = node["verbose"].as<bool>(verbose);
  } else {
    TORCH_CHECK(false, "Unsupported layout type: ", op->type());
  }

  if (op->verbose() && get_rank() == 0) {
    std::cout << "[LayoutOptions] layout options:" << std::endl;
    op->report(std::cout);
  }

  return op;
}

std::shared_ptr<LayoutImpl> LayoutImpl::create(LayoutOptions const& options,
                                               torch::nn::Module* p,
                                               std::string const& name) {
  if (p == nullptr) options->no_backend(true);

  if (options->type() == "slab") {
    return p ? p->register_module(name, SlabLayout(options))
             : SlabLayout(options).ptr();
  } else if (options->type() == "cubed") {
    return p ? p->register_module(name, CubedLayout(options))
             : CubedLayout(options).ptr();
  } else if (options->type() == "cubed-sphere") {
    return p ? p->register_module(name, CubedSphereLayout(options))
             : CubedSphereLayout(options).ptr();
  } else {
    TORCH_CHECK(false, "Unsupported layout type: ", options->type());
  }
}

void LayoutImpl::init_buffers(MeshBlockImpl const* pmb, Variables const& vars,
                              std::vector<std::string> const& names) {
  if (options->verbose()) {
    std::cout << "[Rank " << options->rank() << ":" << options->local_rank()
              << "] Initializing communication buffers...\n";
  }

  // Initialize vectors to size 9 (2D decomposition) with empty tensors
  send_bufs.clear();
  recv_bufs.clear();
  send_bufs.resize(9);
  recv_bufs.resize(9);

  // Iterate over all 2D neighbor directions
  buf_names.clear();

  // only include names that exist in vars
  for (auto name : names) {
    if (vars.count(name) > 0) buf_names.push_back(name);
  }

  // Get my logical location
  auto iloc = loc_of(options->rank());

  for (int x3_offset = -1; x3_offset <= 1; ++x3_offset) {
    for (int x2_offset = -1; x2_offset <= 1; ++x2_offset) {
      // Skip the center (self)
      if (x3_offset == 0 && x2_offset == 0) continue;
      std::tuple<int, int, int> offset(x3_offset, x2_offset, 0);

      int nb = neighbor_rank(iloc, offset);
      // Skip if no neighbor in this direction
      if (nb < 0) continue;

      int bid = get_buffer_id(offset);

      // Get the part indices for this neighbor direction
      auto sub = pmb->part(offset);

      // Get shape by applying indices to tensor
      send_bufs[bid].clear();
      recv_bufs[bid].clear();

      for (auto name : buf_names) {
        auto sub_tensor = vars.at(name).index(sub);

        // Allocate send and receive buffers with same shape
        send_bufs[bid].push_back(torch::empty_like(sub_tensor));
        recv_bufs[bid].push_back(torch::empty_like(sub_tensor));
      }
    }
  }
}

void LayoutImpl::serialize(MeshBlockImpl const* pmb, Variables const& vars) {
  // Iterate over all 2D neighbor directions
  for (int x3_offset = -1; x3_offset <= 1; ++x3_offset) {
    for (int x2_offset = -1; x2_offset <= 1; ++x2_offset) {
      // Skip the center (self)
      if (x3_offset == 0 && x2_offset == 0) continue;

      std::tuple<int, int, int> offset(x3_offset, x2_offset, 0);
      int bid = get_buffer_id(offset);

      // Only serialize if buffer exists
      if (!send_bufs[bid].empty()) {
        // Get the interior part for this direction
        auto sub = pmb->part(offset, /*exterior=*/false);

        // Copy data from mesh to send buffer
        int count = 0;
        for (auto name : buf_names)
          send_bufs[bid][count++].copy_(vars.at(name).index(sub));
      }
    }
  }
}

void LayoutImpl::deserialize(MeshBlockImpl const* pmb, Variables& vars) const {
  // Iterate over all 2D neighbor directions
  for (int x3_offset = -1; x3_offset <= 1; ++x3_offset) {
    for (int x2_offset = -1; x2_offset <= 1; ++x2_offset) {
      // Skip the center (self)
      if (x3_offset == 0 && x2_offset == 0) continue;

      std::tuple<int, int, int> offset(x3_offset, x2_offset, 0);
      int bid = get_buffer_id(offset);

      // Only deserialize if buffer exists
      if (!recv_bufs[bid].empty()) {
        // Get the exterior (ghost zone) part for this direction
        auto sub = pmb->part(offset, /*exterior=*/true);

        // Copy data from receive buffer to mesh ghost zones
        int count = 0;
        for (auto name : buf_names) {
          vars.at(name).index_put_(sub, recv_bufs[bid][count++]);
        }
      }
    }
  }
}

void LayoutImpl::_init_backend() {
  if (options->no_backend()) return;

  if (options->verbose()) {
    std::cout << "[Rank " << options->rank() << ":" << options->local_rank()
              << "] Initializing distributed environment...\n";
  }

  // 1. Build the store
  c10d::TCPStoreOptions store_op;

  store_op.port = options->master_port();
  store_op.numWorkers = options->world_size();
  store_op.isServer = is_root();

  store = at::make_intrusive<c10d::TCPStore>(options->master_addr(), store_op);

  // 2. Create ProcessGroup based on backend
  if (options->backend() == "gloo") {
    _init_gloo();
  } else if (options->backend() == "nccl") {
    _init_nccl();
  } else {
    throw std::runtime_error("Unsupported BACKEND=" + options->backend());
  }

  if (options->verbose()) {
    std::cout << "[Rank " << options->rank() << ":" << options->local_rank()
              << "] Distributed environment initialized with backend="
              << options->backend() << ", world_size=" << options->world_size()
              << "\n";
  }
}

void LayoutImpl::_init_gloo() {
  if (options->verbose()) {
    std::cout << "[Rank " << options->rank() << ":" << options->local_rank()
              << "] Using Gloo backend on CPU\n";
  }

  auto opts = c10d::ProcessGroupGloo::Options::create();
  opts->devices.push_back(c10d::ProcessGroupGloo::createDefaultDevice());

  pg = std::make_shared<c10d::ProcessGroupGloo>(store, options->rank(),
                                                options->world_size(), opts);
}

__attribute__((weak)) void LayoutImpl::_init_nccl() {}

}  // namespace snap
