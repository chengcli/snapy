// yaml
#include <sys/utsname.h>
#include <yaml-cpp/yaml.h>

#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <random>
#include <sstream>
#include <thread>

// base
#include <configure.h>

#ifdef USE_CUDA
#include <ATen/cuda/CUDAEvent.h>
#include <c10/cuda/CUDAStream.h>
#endif

// snap
#include <snap/mesh/meshblock.hpp>
#include <snap/utils/log.hpp>

#include "cubed_sphere_layout.hpp"
#include "layout.hpp"

namespace snap {

namespace {

int random_master_port() {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<int> dist(29500, 29600);
  return dist(gen);
}

std::string default_backend() {
#ifdef USE_UCX
  struct utsname system_info;
  if (uname(&system_info) == 0 && std::string(system_info.sysname) == "Darwin")
    return "gloo";
  return "ucx";
#else
  return "gloo";
#endif
}

struct RemoteExchangeOp {
  LayoutImpl* layout;
  int remote_process;
  int local_block;
  int remote_local_block;
  int buffer_id;
  int send_tag;
  int recv_tag;
  std::tuple<int, int, int> offset;
  std::tuple<int, int, int> peer_offset;
};

std::mutex g_process_comm_mutex;

std::pair<int, int> exchange_dz_bounds(LayoutImpl const& layout,
                                       SyncOptions const& opts) {
  if (layout.num_exchange_buffers() < 27) return {0, 0};
  return {opts.dz_min(), opts.dz_max()};
}

std::string exchange_buffer_key(SyncOptions const& opts,
                                Variables const& vars) {
  std::ostringstream key;
  key << opts.dim() << ':' << opts.phyid() << ':' << opts.type() << ':'
      << opts.cross_panel_only() << ':' << opts.skip_corner() << ':'
      << opts.interpolate();
  for (auto const& [name, _] : vars) {
    key << ':' << name;
  }
  return key.str();
}

bool buffer_matches(torch::Tensor const& buffer, torch::Tensor const& source) {
  return buffer.defined() && buffer.sizes() == source.sizes() &&
         buffer.scalar_type() == source.scalar_type() &&
         buffer.device() == source.device();
}

}  // namespace

LayoutOptionsImpl::LayoutOptionsImpl() {
  backend(get_env("BACKEND", default_backend()));

  // These enrionment variables will be set by torch.distributed.launch
  // Override by them if they are present
  auto process_rank_env = get_env("PROCESS_RANK", get_env("RANK", "0"));
  auto process_world_size_env =
      get_env("PROCESS_WORLD_SIZE", get_env("WORLD_SIZE", "1"));
  auto world_size_env = get_env("WORLD_SIZE", process_world_size_env);
  int process_world_size_value = std::stoi(process_world_size_env);
  int world_size_value = std::stoi(world_size_env);
  master_addr(get_env("MASTER_ADDR", "127.0.0.1"));
  auto master_port_env = std::getenv("MASTER_PORT");
  if (master_port_env) {
    master_port(std::stoi(master_port_env));
  } else {
    TORCH_CHECK(process_world_size_value == 1 && world_size_value == 1,
                "MASTER_PORT must be set for multi-process runs "
                "(PROCESS_WORLD_SIZE=",
                process_world_size_value, ", WORLD_SIZE=", world_size_value,
                ") so all ranks rendezvous on the same TCPStore");
    master_port(random_master_port());
  }
  process_rank(std::stoi(process_rank_env));
  rank(std::stoi(get_env("RANK", process_rank_env)));
  local_rank(std::stoi(get_env("LOCAL_RANK", "0")));
  process_world_size(process_world_size_value);
  world_size(world_size_value);
  device_id(std::stoi(get_env("DEVICE_ID", "-1")));
  device(get_env("DEVICE", "cpu"));
}

LayoutOptions LayoutOptionsImpl::from_yaml(std::string const& filename,
                                           bool verbose) {
  auto op = LayoutOptionsImpl::create();
  auto config = YAML::LoadFile(filename);

  if (!config["distribute"]) return op;

  auto node = config["distribute"];

  op->type() = node["layout"].as<std::string>("slab");
  op->py(node["nb3"].as<int>(1));
  op->px(node["nb2"].as<int>(1));
  op->pz(node["nb1"].as<int>(1));
  op->backend() = get_env("BACKEND", op->backend());
  op->device() = get_env("DEVICE", "cpu");
  op->verbose() = node["verbose"].as<bool>(verbose);

  if (op->verbose()) op->report(SINFO(LayoutOptions));

  return op;
}

std::shared_ptr<LayoutImpl> LayoutImpl::create(LayoutOptions const& options,
                                               MeshBlockImpl* p,
                                               std::string const& name) {
  (void)name;

  std::shared_ptr<LayoutImpl> pl;
  if (options->type() == "slab") {
    pl = std::make_shared<SlabLayoutImpl>(options, p);
  } else if (options->type() == "cubed") {
    pl = std::make_shared<CubedLayoutImpl>(options, p);
  } else if (options->type() == "cubed-sphere") {
    pl = std::make_shared<CubedSphereLayoutImpl>(options, p);
  } else {
    TORCH_CHECK(false, "Unsupported layout type: ", options->type());
  }

  return pl;
}

LayoutImpl::~LayoutImpl() = default;

void LayoutImpl::connect_local_layouts(
    std::vector<LayoutImpl*> const& local_layouts) {
  std::map<int, LayoutImpl*> by_rank;
  for (auto* layout : local_layouts) {
    TORCH_CHECK(layout != nullptr, "cannot connect a null local layout");
    layout->_incoming_local.clear();
    layout->_outgoing_local.clear();
    layout->_local_ghost_state = {};
    by_rank.emplace(layout->options->rank(), layout);
  }

  SyncOptions topology_opts;
  for (auto* source : local_layouts) {
    int rank = source->options->rank();
    auto iloc = source->loc_of(rank);
    int dz_min = source->num_exchange_buffers() < 27 ? 0 : -1;
    int dz_max = source->num_exchange_buffers() < 27 ? 0 : 1;

    for (int dz = dz_min; dz <= dz_max; ++dz) {
      for (int dy = -1; dy <= 1; ++dy) {
        for (int dx = -1; dx <= 1; ++dx) {
          if (dz == 0 && dy == 0 && dx == 0) continue;
          auto offset = source->_remap_exchange_offset(iloc, dy, dx, dz);
          int neighbor = source->neighbor_rank(iloc, offset);
          if (neighbor < 0 || neighbor == rank) continue;
          if (source->options->type() == "cubed-sphere" &&
              std::abs(dy) + std::abs(dx) > 1 &&
              std::get<2>(iloc) != std::get<2>(source->loc_of(neighbor))) {
            continue;
          }
          if (source->options->owner_process_rank(neighbor) !=
              source->options->process_rank()) {
            continue;
          }

          auto target_it = by_rank.find(neighbor);
          TORCH_CHECK(target_it != by_rank.end(),
                      "missing local layout for block rank ", neighbor);
          auto* target = target_it->second;
          auto peer_offset = source->_peer_exchange_offset(
              neighbor, rank, topology_opts, offset);
          auto connection = std::make_shared<LocalGhostConnection>();
          connection->source_rank = rank;
          connection->target_rank = neighbor;
          connection->source_buffer_id = get_buffer_id(offset);
          connection->target_buffer_id = get_buffer_id(peer_offset);
          connection->source_offset = {dy, dx, dz};
#ifdef USE_CUDA
          if (source->options->device() == "cuda") {
            connection->ready_event =
                std::make_shared<at::cuda::CUDAEvent>();
            connection->consumed_event =
                std::make_shared<at::cuda::CUDAEvent>();
          }
#endif
          source->_outgoing_local.push_back(connection);
          target->_incoming_local.push_back(std::move(connection));
        }
      }
    }
  }
}

void LayoutImpl::_prepare_local_exchange(MeshBlockImpl const* pmb,
                                         SyncOptions const& opts) {
  if (options->blocks_per_process() <= 1) return;
  TORCH_CHECK(pmb != nullptr,
              "local exchange requires an owning MeshBlock pointer");

#ifdef USE_CUDA
  std::optional<c10::cuda::CUDAStream> current_stream;
  if (options->device() == "cuda") {
    current_stream.emplace(c10::cuda::getCurrentCUDAStream());
  }
#endif

  auto& state = _local_ghost_state;
  state.generation += 1;
  state.expected_mask = 0;
  state.arrived_mask = 0;

  std::vector<std::pair<std::shared_ptr<LocalGhostConnection>, std::uint64_t>>
      published;
  published.reserve(_outgoing_local.size());
  for (auto const& connection : _outgoing_local) {
    auto [dy, dx, dz] = connection->source_offset;
    bool active = dy >= opts.dy_min() && dy <= opts.dy_max() &&
                  dx >= opts.dx_min() && dx <= opts.dx_max() &&
                  dz >= opts.dz_min() && dz <= opts.dz_max();
    if (active && opts.skip_corner() &&
        std::abs(dy) + std::abs(dx) + std::abs(dz) > 1) {
      active = false;
    }
    if (active && options->type() == "cubed-sphere" &&
        opts.cross_panel_only()) {
      active = std::get<2>(loc_of(connection->source_rank)) !=
               std::get<2>(loc_of(connection->target_rank));
    }
    LocalGhostMessage message;
    message.generation = state.generation;
    message.source_buffer_id = connection->source_buffer_id;
    message.target_buffer_id = connection->target_buffer_id;
    message.active = active;
    if (active) {
      message.buffers = pmb->send_bufs.at(connection->source_buffer_id);
    }
#ifdef USE_CUDA
    if (current_stream && connection->ready_event) {
      connection->ready_event->record(*current_stream);
    }
#endif
    auto ticket = connection->queue.wait_push(std::move(message));
    published.emplace_back(connection, ticket);
  }

  std::size_t processed = 0;
  while (processed != _incoming_local.size()) {
    bool made_progress = false;
    for (auto const& connection : _incoming_local) {
      auto const* message = connection->queue.front();
      if (message == nullptr || message->generation > state.generation) continue;
      TORCH_CHECK(message->generation == state.generation,
                  "stale local ghost message for block ", options->rank(),
                  ": expected generation ", state.generation, " but received ",
                  message->generation);

      bool consumed = connection->queue.try_consume(
          [&](LocalGhostMessage& current) {
            TORCH_CHECK(current.source_buffer_id ==
                            connection->source_buffer_id &&
                            current.target_buffer_id ==
                                connection->target_buffer_id,
                        "local ghost connection metadata mismatch");
            if (current.active) {
              auto bit = std::uint32_t{1} << current.target_buffer_id;
              state.expected_mask |= bit;
#ifdef USE_CUDA
              if (current_stream && connection->ready_event) {
                connection->ready_event->block(*current_stream);
              }
#endif
              auto& destination =
                  pmb->recv_bufs.at(current.target_buffer_id);
              TORCH_CHECK(destination.size() == current.buffers.size(),
                          "local exchange tensor-count mismatch from rank ",
                          connection->source_rank, " to rank ",
                          connection->target_rank);
              for (std::size_t n = 0; n < current.buffers.size(); ++n) {
                TORCH_CHECK(destination[n].numel() ==
                                current.buffers[n].numel(),
                            "local exchange size mismatch from rank ",
                            connection->source_rank, " to rank ",
                            connection->target_rank);
                destination[n].view({-1}).copy_(
                    current.buffers[n].reshape({-1}));
              }
              state.arrived_mask |= bit;
            }
#ifdef USE_CUDA
            if (current_stream && connection->consumed_event) {
              connection->consumed_event->record(*current_stream);
            }
#endif
          });
      if (consumed) {
        processed += 1;
        made_progress = true;
      }
    }
    if (!made_progress) std::this_thread::yield();
  }

  TORCH_CHECK(state.ready(),
              "local ghost exchange did not receive every region");
  for (auto const& [connection, ticket] : published) {
    while (!connection->queue.consumed(ticket)) {
      std::this_thread::yield();
    }
#ifdef USE_CUDA
    if (current_stream && connection->consumed_event) {
      connection->consumed_event->block(*current_stream);
    }
#endif
  }
}

std::tuple<int, int, int> LayoutImpl::_remap_exchange_offset(
    std::tuple<int, int, int> iloc, int dy, int dx, int dz) const {
  int dx_sgn = 1;
  int dy_sgn = 1;
  int dz_sgn = 1;

  if (options->periodic_x() && options->px() == 2 && std::get<0>(iloc) == 0) {
    dx_sgn = -1;
  }

  if (options->periodic_y() && options->py() == 2 && std::get<1>(iloc) == 0) {
    dy_sgn = -1;
  }

  if (options->periodic_z() && options->pz() == 2 && std::get<2>(iloc) == 0) {
    dz_sgn = -1;
  }

  return {dy_sgn * dy, dx_sgn * dx, dz_sgn * dz};
}

std::tuple<int, int, int> LayoutImpl::_peer_exchange_offset(
    int peer_rank, int target_rank, SyncOptions const& opts,
    std::tuple<int, int, int> offset) const {
  (void)peer_rank;
  (void)target_rank;
  (void)opts;
  auto [dy, dx, dz] = offset;
  return {-dy, -dx, -dz};
}

void LayoutImpl::serialize(MeshBlockImpl const* pmb, Variables& vars,
                           SyncOptions const& opts) {
  if (options->verbose()) {
    SINFO(Layout) << "serializing data into send buffers\n";
  }

  // Get my logical location
  auto iloc = loc_of(options->rank());

  // Iterate over all 3D neighbor directions
  auto [dz_min, dz_max] = exchange_dz_bounds(*this, opts);
  int dy_min = opts.dy_min();
  int dy_max = opts.dy_max();
  int dx_min = opts.dx_min();
  int dx_max = opts.dx_max();
  auto& cache = pmb->exchange_buffer_cache[exchange_buffer_key(opts, vars)];
  cache.send.resize(num_exchange_buffers());
  cache.recv.resize(num_exchange_buffers());

  for (int dz = dz_min; dz <= dz_max; ++dz)
    for (int dy = dy_min; dy <= dy_max; ++dy)
      for (int dx = dx_min; dx <= dx_max; ++dx) {
        // Skip the center (self)
        if (dz == 0 && dy == 0 && dx == 0) continue;
        // In 3D, skip_corner keeps only face-normal directions (one non-zero
        // component)
        if (opts.skip_corner() &&
            std::abs(dz) + std::abs(dy) + std::abs(dx) > 1)
          continue;
        if (pmb->options->is_physical_boundary(dy, dx, dz)) continue;

        std::tuple<int, int, int> offset(dy, dx, dz);
        int nb = neighbor_rank(iloc, offset);
        if (nb < 0) continue;  // no neighbor

        // Get the interior part for this direction
        auto sub = pmb->part(offset, PartOptions().exterior(false));

        // Copy data from mesh to send buffer
        int bid = get_buffer_id(offset);
        int count = 0;
        auto& cached_send = cache.send[bid];
        auto& cached_recv = cache.recv[bid];
        cached_send.resize(vars.size());
        cached_recv.resize(vars.size());
        for (auto& [name, var] : vars) {
          auto source = var.index(sub);
          if (!buffer_matches(cached_send[count], source)) {
            cached_send[count] = torch::empty(source.sizes(), source.options());
          }
          cached_send[count].copy_(source);

          if (!buffer_matches(cached_recv[count], cached_send[count])) {
            cached_recv[count] = torch::empty(cached_send[count].sizes(),
                                              cached_send[count].options());
          }
          count++;
        }
        pmb->send_bufs[bid] = cached_send;
        pmb->recv_bufs[bid] = cached_recv;
      }
}

void LayoutImpl::launch_exchange(MeshBlockImpl const* pmb,
                                 SyncOptions const& opts,
                                 std::vector<CommWorkPtr>& works) {
  _prepare_local_exchange(pmb, opts);

  exchange_remote(pmb, opts, works);
}

void LayoutImpl::exchange_remote(MeshBlockImpl const* pmb,
                                 SyncOptions const& opts,
                                 std::vector<CommWorkPtr>& works) {
  TORCH_CHECK(owner() != nullptr,
              "[Layout:exchange_remote] layout has no owning MeshBlock");
  TORCH_CHECK(pmb != nullptr,
              "[Layout:exchange_remote] MeshBlock pointer is null");

  if (options->verbose()) {
    SINFO(Layout) << "performing communication\n";
  }

  // Get my rank
  auto rank = options->rank();

  // Get my logical location
  auto iloc = loc_of(rank);

  auto [dz_min, dz_max] = exchange_dz_bounds(*this, opts);
  int dy_min = opts.dy_min();
  int dy_max = opts.dy_max();
  int dx_min = opts.dx_min();
  int dx_max = opts.dx_max();
  int dx_sgn = 1;
  int dy_sgn = 1;
  int dz_sgn = 1;

  // swap the order of first block for periodic condition
  if (options->periodic_x() && options->px() == 2 && std::get<0>(iloc) == 0) {
    dx_sgn = -1;
  }

  if (options->periodic_y() && options->py() == 2 && std::get<1>(iloc) == 0) {
    dy_sgn = -1;
  }

  if (options->periodic_z() && options->pz() == 2 && std::get<2>(iloc) == 0) {
    dz_sgn = -1;
  }

  std::vector<RemoteExchangeOp> remote_ops;

  for (int dz_ = dz_min; dz_ <= dz_max; ++dz_)
    for (int dy_ = dy_min; dy_ <= dy_max; ++dy_)
      for (int dx_ = dx_min; dx_ <= dx_max; ++dx_) {
        int dz = dz_sgn * dz_;
        int dy = dy_sgn * dy_;
        int dx = dx_sgn * dx_;

        // skip the center (self)
        if (dz == 0 && dy == 0 && dx == 0) continue;
        if (opts.skip_corner() &&
            std::abs(dz) + std::abs(dy) + std::abs(dx) > 1)
          continue;
        if (pmb->options->is_physical_boundary(dy, dx, dz)) continue;

        std::tuple<int, int, int> offset(dy, dx, dz);
        int nb = neighbor_rank(iloc, offset);
        if (nb < 0) continue;  // no neighbor

        int r = get_buffer_id(offset);
        int remote_process = options->owner_process_rank(nb);
        bool is_remote = remote_process != options->process_rank();

        if (is_remote) {
          int remote_local_block = options->local_block_index(nb);
          int local_block = options->local_block_index(rank);
          auto peer_offset = _peer_exchange_offset(nb, rank, opts, offset);
          int send_id =
              make_comm_tag(remote_local_block, peer_offset, opts.phyid());
          int recv_id = make_comm_tag(local_block, offset, opts.phyid());
          remote_ops.push_back({this, remote_process, local_block,
                                remote_local_block, r, send_id, recv_id, offset,
                                peer_offset});
        } else if (nb == rank) {  // self-send
          int r1 = get_buffer_id(std::tuple<int, int, int>(-dy, -dx, -dz));
          for (int n = 0; n < pmb->recv_bufs[r].size(); ++n)
            pmb->recv_bufs[r1][n].copy_(pmb->send_bufs[r][n]);
        }
      }

  if (remote_ops.empty()) return;
  TORCH_CHECK(has_process_group(),
              "[Layout:exchange_remote] remote communication requires an "
              "initialized process group");

  std::lock_guard<std::mutex> lock(g_process_comm_mutex);
  bool coalescing = comm->supports_coalescing();
  if (coalescing) {
    comm->start_coalescing();
  }
  for (auto const& op : remote_ops) {
    auto send_work = comm->send(pmb->send_bufs[op.buffer_id], op.remote_process,
                                op.send_tag);
    if (send_work) {
      works.push_back(send_work);
    }
    auto recv_work = comm->recv(pmb->recv_bufs[op.buffer_id], op.remote_process,
                                op.recv_tag);
    if (recv_work) {
      works.push_back(recv_work);
    }
  }
  if (coalescing) {
    auto coalesced_work = comm->end_coalescing();
    if (coalesced_work) {
      works.push_back(coalesced_work);
    }
  }
}

void LayoutImpl::deserialize(MeshBlockImpl const* pmb, Variables& vars,
                             SyncOptions const& opts) const {
  if (options->verbose()) {
    SINFO(Layout) << "deserializing data from receive buffers\n";
  }

  // Get my logical location
  auto iloc = loc_of(options->rank());

  auto [dz_min, dz_max] = exchange_dz_bounds(*this, opts);
  int dy_min = opts.dy_min();
  int dy_max = opts.dy_max();
  int dx_min = opts.dx_min();
  int dx_max = opts.dx_max();

  // Iterate over all 3D neighbor directions
  for (int dz = dz_min; dz <= dz_max; ++dz)
    for (int dy = dy_min; dy <= dy_max; ++dy)
      for (int dx = dx_min; dx <= dx_max; ++dx) {
        // Skip the center (self)
        if (dz == 0 && dy == 0 && dx == 0) continue;
        if (opts.skip_corner() &&
            std::abs(dz) + std::abs(dy) + std::abs(dx) > 1)
          continue;
        if (pmb->options->is_physical_boundary(dy, dx, dz)) continue;

        std::tuple<int, int, int> offset(dy, dx, dz);
        int nb = neighbor_rank(iloc, offset);
        if (nb < 0) continue;  // no neighbor

        // Get the exterior (ghost zone) part for this direction
        auto sub = pmb->part(offset, PartOptions().exterior(true));

        // Copy data from receive buffer to mesh ghost zones
        int bid = get_buffer_id(offset);
        int count = 0;
        for (auto& [name, var] : vars) {
          var.index_put_(sub, pmb->recv_bufs[bid][count++]);
        }
      }
}

void LayoutImpl::fill_corners(MeshBlockImpl const* pmb, Variables& vars) const {
  auto sub_left = pmb->part({0, -1, 0}, PartOptions().exterior(true));
  auto sub_right = pmb->part({0, +1, 0}, PartOptions().exterior(true));
  auto sub_bot = pmb->part({-1, 0, 0}, PartOptions().exterior(true));
  auto sub_top = pmb->part({+1, 0, 0}, PartOptions().exterior(true));

  // Fill-in left-bot inter-panel corners
  std::tuple<int, int, int> corner(/*dy=*/-1, /*dx=*/-1, 0);
  auto sub = pmb->part(corner, PartOptions().exterior(true));
  for (auto& [name, var] : vars) {
    auto var_left = var.index(sub_left).select(-3, 0).unsqueeze(-3);
    auto var_bot = var.index(sub_bot).select(-2, 0).unsqueeze(-2);
    var.index_put_(sub, 0.5 * (var_left + var_bot));
  }

  // Fill-in right-bot inter-panel corners
  corner = std::tuple<int, int, int>(/*dy=*/-1, /*dx=*/1, 0);
  sub = pmb->part(corner, PartOptions().exterior(true));
  for (auto& [name, var] : vars) {
    auto var_right = var.index(sub_right).select(-3, 0).unsqueeze(-3);
    auto var_bot = var.index(sub_bot).select(-2, -1).unsqueeze(-2);
    var.index_put_(sub, 0.5 * (var_right + var_bot));
  }

  // Fill-in left-top inter-panel corners
  corner = std::tuple<int, int, int>(/*dy=*/1, /*dx=*/-1, 0);
  sub = pmb->part(corner, PartOptions().exterior(true));
  for (auto& [name, var] : vars) {
    auto var_left = var.index(sub_left).select(-3, -1).unsqueeze(-3);
    auto var_top = var.index(sub_top).select(-2, 0).unsqueeze(-2);
    var.index_put_(sub, 0.5 * (var_left + var_top));
  }

  // Fill-in right-top inter-panel corners
  corner = std::tuple<int, int, int>(/*dy=*/1, /*dx=*/1, 0);
  sub = pmb->part(corner, PartOptions().exterior(true));
  for (auto& [name, var] : vars) {
    auto var_right = var.index(sub_right).select(-3, -1).unsqueeze(-3);
    auto var_top = var.index(sub_top).select(-2, -1).unsqueeze(-2);
    var.index_put_(sub, 0.5 * (var_right + var_top));
  }
}

void LayoutImpl::finalize(MeshBlockImpl const* pmb, Variables& vars,
                          SyncOptions const& opts,
                          std::vector<CommWorkPtr>& works) {
  // Wait for all operations to complete
  for (auto& work : works) {
    if (work) {
      work->wait();
    }
  }
  // Deserialize received data into ghost zones
  deserialize(pmb, vars, opts);

  // Fill corners
  if (opts.skip_corner() && !opts.cross_panel_only()) {
    fill_corners(pmb, vars);
  }

  // Completed point-to-point work is sufficient; a global barrier would
  // serialize otherwise independent exchanges.
  works.clear();
}

void LayoutImpl::_init_process_group() {
  if (!use_process_group()) return;
  comm = ProcessGroupContext::create(options);
}

}  // namespace snap
