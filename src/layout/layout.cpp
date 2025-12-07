// yaml
#include <yaml-cpp/yaml.h>

// base
#include <configure.h>  // gloo

// torch
#include <torch/csrc/distributed/c10d/ProcessGroupGloo.hpp>
#include <torch/csrc/distributed/c10d/TCPStore.hpp>

// snap
#include <snap/mesh/meshblock.hpp>

#include "cubed_sphere_layout.hpp"
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
  op->px(node["nb3"].as<int>(1));
  op->py(node["nb2"].as<int>(1));
  op->pz(node["nb1"].as<int>(1));
  op->backend() = node["backend"].as<std::string>("gloo");
  op->verbose() = node["verbose"].as<bool>(verbose);

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

  std::shared_ptr<LayoutImpl> pl;
  if (options->type() == "slab") {
    pl = p ? p->register_module(name, SlabLayout(options))
           : SlabLayout(options).ptr();
    pl->send_bufs.resize(9);
    pl->recv_bufs.resize(9);
  } else if (options->type() == "cubed") {
    pl = p ? p->register_module(name, CubedLayout(options))
           : CubedLayout(options).ptr();
    pl->send_bufs.resize(27);
    pl->recv_bufs.resize(27);
  } else if (options->type() == "cubed-sphere") {
    pl = p ? p->register_module(name, CubedSphereLayout(options))
           : CubedSphereLayout(options).ptr();
    pl->send_bufs.resize(9);
    pl->recv_bufs.resize(9);
  } else {
    TORCH_CHECK(false, "Unsupported layout type: ", options->type());
  }

  return pl;
}

void LayoutImpl::serialize(MeshBlockImpl const* pmb, Variables& vars,
                           SyncOptions opts) {
  if (options->verbose() && is_root()) {
    std::cout << "[Layout] serializing data into send buffers\n";
  }

  // Get my logical location
  auto iloc = loc_of(options->rank());

  // Iterate over all 2D neighbor directions
  int x3_omin = opts.x3_offset_min();
  int x3_omax = opts.x3_offset_max();
  int x2_omin = opts.x2_offset_min();
  int x2_omax = opts.x2_offset_max();

  for (int x3_offset = x3_omin; x3_offset <= x3_omax; ++x3_offset)
    for (int x2_offset = x2_omin; x2_offset <= x2_omax; ++x2_offset) {
      // Skip the center (self)
      if (x3_offset == 0 && x2_offset == 0) continue;
      if (opts.skip_corner() && std::abs(x3_offset) + std::abs(x2_offset) == 2)
        continue;

      std::tuple<int, int, int> offset(x3_offset, x2_offset, 0);
      int nb = neighbor_rank(iloc, offset);
      if (nb < 0) continue;  // no neighbor

      // Get the interior part for this direction
      auto sub = pmb->part(offset, PartOptions().exterior(false));

      // Copy data from mesh to send buffer
      int bid = get_buffer_id(offset);
      int count = 0;
      send_bufs[bid].resize(vars.size());
      recv_bufs[bid].resize(vars.size());
      for (auto& [name, var] : vars) {
        send_bufs[bid][count] = var.index(sub).clone();
        recv_bufs[bid][count] = torch::empty_like(send_bufs[bid][count]);
        count++;
      }
    }
}

void LayoutImpl::deserialize(MeshBlockImpl const* pmb, Variables& vars,
                             SyncOptions opts) const {
  if (options->verbose() && is_root()) {
    std::cout << "[Layout] deserializing data from receive buffers\n";
  }

  // Get my logical location
  auto iloc = loc_of(options->rank());

  int x3_omin = opts.x3_offset_min();
  int x3_omax = opts.x3_offset_max();
  int x2_omin = opts.x2_offset_min();
  int x2_omax = opts.x2_offset_max();

  // Iterate over all 2D neighbor directions
  for (int x3_offset = x3_omin; x3_offset <= x3_omax; ++x3_offset)
    for (int x2_offset = x2_omin; x2_offset <= x2_omax; ++x2_offset) {
      // Skip the center (self)
      if (x3_offset == 0 && x2_offset == 0) continue;
      if (opts.skip_corner() && std::abs(x3_offset) + std::abs(x2_offset) == 2)
        continue;

      std::tuple<int, int, int> offset(x3_offset, x2_offset, 0);
      int nb = neighbor_rank(iloc, offset);
      if (nb < 0) continue;  // no neighbor

      // Get the exterior (ghost zone) part for this direction
      auto sub = pmb->part(offset, PartOptions().exterior(true));

      // Copy data from receive buffer to mesh ghost zones
      int bid = get_buffer_id(offset);
      int count = 0;
      for (auto& [name, var] : vars) {
        var.index_put_(sub, recv_bufs[bid][count++]);
      }
    }
}

void LayoutImpl::_init_backend() {
  if (options->no_backend()) return;

  if (options->verbose()) {
    std::cout << "[Rank " << options->rank() << ":" << options->local_rank()
              << "] Initializing distributed environment\n";
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
