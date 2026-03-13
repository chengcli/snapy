// yaml
#include <condition_variable>
#include <exception>
#include <map>
#include <yaml-cpp/yaml.h>

// base
#include <configure.h>  // gloo and nccl

// snap
#include <snap/mesh/meshblock.hpp>
#include <snap/utils/log.hpp>

#include "cubed_sphere_layout.hpp"
#include "layout.hpp"

namespace snap {

namespace {

struct LocalExchangeKey {
  int process_rank;
  int blocks_per_process;
  int dim;
  int phyid;
  int type;
  bool cross_panel_only;
  bool skip_corner;
  bool interpolate;
  std::string layout_type;

  bool operator<(LocalExchangeKey const& other) const {
    return std::tie(process_rank, blocks_per_process, dim, phyid, type,
                    cross_panel_only, skip_corner, interpolate, layout_type) <
           std::tie(other.process_rank, other.blocks_per_process, other.dim,
                    other.phyid, other.type, other.cross_panel_only,
                    other.skip_corner, other.interpolate, other.layout_type);
  }
};

struct LocalExchangeState {
  int generation = 0;
  int arrived = 0;
  int released = 0;
  bool ready = false;
  std::exception_ptr error;
};

std::mutex g_local_exchange_mutex;
std::condition_variable g_local_exchange_cv;
std::map<std::pair<int, int>, LayoutImpl*> g_local_layouts;
std::map<LocalExchangeKey, LocalExchangeState> g_local_exchange_states;
std::mutex g_process_comm_mutex;

std::vector<LayoutImpl*> local_layouts_for(LayoutImpl const& layout) {
  std::vector<LayoutImpl*> layouts(layout.options->blocks_per_process(), nullptr);

  for (auto const& [key, local_layout] : g_local_layouts) {
    if (key.first == layout.options->process_rank()) {
      TORCH_CHECK(key.second >= 0 &&
                      key.second < layout.options->blocks_per_process(),
                  "invalid local block index ", key.second,
                  " for process-local layout registry");
      layouts[key.second] = local_layout;
    }
  }

  for (int i = 0; i < layouts.size(); ++i) {
    TORCH_CHECK(layouts[i] != nullptr,
                "missing process-local layout registration for process ",
                layout.options->process_rank(), " local block ", i);
  }

  return layouts;
}

LocalExchangeKey make_local_exchange_key(LayoutImpl const& layout,
                                         SyncOptions const& opts) {
  return LocalExchangeKey{
      layout.options->process_rank(),
      layout.options->blocks_per_process(),
      opts.dim(),
      opts.phyid(),
      opts.type(),
      opts.cross_panel_only(),
      opts.skip_corner(),
      opts.interpolate(),
      layout.options->type(),
  };
}

}  // namespace

LayoutOptionsImpl::LayoutOptionsImpl() {
  // These enrionment variables will be set by torch.distributed.launch
  // Override by them if they are present
  auto process_rank_env = get_env("PROCESS_RANK", get_env("RANK", "0"));
  auto process_world_size_env =
      get_env("PROCESS_WORLD_SIZE", get_env("WORLD_SIZE", "1"));
  master_addr(get_env("MASTER_ADDR", "127.0.0.1"));
  master_port(std::stoi(get_env("MASTER_PORT", "29501")));
  process_rank(std::stoi(process_rank_env));
  rank(std::stoi(get_env("RANK", process_rank_env)));
  local_rank(std::stoi(get_env("LOCAL_RANK", "0")));
  process_world_size(std::stoi(process_world_size_env));
  world_size(std::stoi(get_env("WORLD_SIZE", process_world_size_env)));
  device_id(std::stoi(get_env("DEVICE_ID", "-1")));
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
  op->backend() = get_env("BACKEND", node["backend"].as<std::string>("gloo"));
  op->verbose() = node["verbose"].as<bool>(verbose);

  if (op->verbose()) op->report(SINFO(LayoutOptions));

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

  if (!options->no_backend()) {
    std::lock_guard<std::mutex> lock(g_local_exchange_mutex);
    g_local_layouts[{options->process_rank(),
                     options->local_block_index(options->rank())}] = pl.get();
  }

  return pl;
}

LayoutImpl::~LayoutImpl() {
  if (options == nullptr || options->no_backend()) return;

  std::lock_guard<std::mutex> lock(g_local_exchange_mutex);
  g_local_layouts.erase(
      {options->process_rank(), options->local_block_index(options->rank())});
}

void LayoutImpl::_prepare_local_exchange(MeshBlockImpl const* pmb,
                                         SyncOptions const& opts) {
  if (options->blocks_per_process() <= 1) return;

  auto key = make_local_exchange_key(*this, opts);
  int expected = options->blocks_per_process();
  std::unique_lock<std::mutex> lock(g_local_exchange_mutex);
  auto& state = g_local_exchange_states[key];
  int generation = state.generation;

  state.arrived += 1;
  if (state.arrived == expected) {
    auto layouts = local_layouts_for(*this);
    lock.unlock();
    std::exception_ptr error;
    try {
      _copy_local_exchange_buffers(layouts, opts);
    } catch (...) {
      error = std::current_exception();
    }
    lock.lock();
    state.error = error;
    state.ready = true;
    state.released = expected;
    g_local_exchange_cv.notify_all();
  } else {
    g_local_exchange_cv.wait(lock, [&]() {
      return state.ready && state.generation == generation;
    });
  }

  if (state.error != nullptr) {
    auto error = state.error;
    state.released -= 1;
    if (state.released == 0) {
      state.arrived = 0;
      state.ready = false;
      state.error = nullptr;
      state.generation += 1;
      g_local_exchange_cv.notify_all();
    } else {
      g_local_exchange_cv.wait(lock, [&]() {
        return state.generation != generation;
      });
    }
    std::rethrow_exception(error);
  }

  state.released -= 1;
  if (state.released == 0) {
    state.arrived = 0;
    state.ready = false;
    state.error = nullptr;
    state.generation += 1;
    g_local_exchange_cv.notify_all();
  } else {
    g_local_exchange_cv.wait(lock, [&]() {
      return state.generation != generation;
    });
  }
}

std::tuple<int, int, int> LayoutImpl::_remap_exchange_offset(
    std::tuple<int, int, int> iloc, int dy, int dx) const {
  int dx_sgn = 1;
  int dy_sgn = 1;

  if (options->periodic_x() && options->px() == 2 && std::get<0>(iloc) == 0) {
    dx_sgn = -1;
  }

  if (options->periodic_y() && options->py() == 2 && std::get<1>(iloc) == 0) {
    dy_sgn = -1;
  }

  return {dy_sgn * dy, dx_sgn * dx, 0};
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

void LayoutImpl::_copy_local_exchange_buffers(
    std::vector<LayoutImpl*> const& layouts, SyncOptions const& opts) const {
  for (auto* layout : layouts) {
    auto rank = layout->options->rank();
    auto iloc = layout->loc_of(rank);

    for (int dy_ = opts.dy_min(); dy_ <= opts.dy_max(); ++dy_) {
      for (int dx_ = opts.dx_min(); dx_ <= opts.dx_max(); ++dx_) {
        if (dy_ == 0 && dx_ == 0) continue;
        if (opts.skip_corner() && std::abs(dy_) + std::abs(dx_) == 2) continue;

        auto offset = layout->_remap_exchange_offset(iloc, dy_, dx_);
        int nb = layout->neighbor_rank(iloc, offset);
        if (nb < 0 || nb == rank) continue;
        if (layout->options->owner_process_rank(nb) !=
            layout->options->process_rank()) {
          continue;
        }

        int bid = get_buffer_id(offset);
        auto peer = layouts.at(layout->options->local_block_index(nb));
        auto peer_offset = layout->_peer_exchange_offset(nb, rank, opts, offset);
        int peer_bid = get_buffer_id(peer_offset);

        for (int n = 0; n < layout->send_bufs[bid].size(); ++n) {
          TORCH_CHECK(peer->recv_bufs[peer_bid][n].numel() ==
                          layout->send_bufs[bid][n].numel(),
                      "local exchange size mismatch from rank ", rank,
                      " to rank ", nb, " send_offset=(",
                      std::get<0>(offset), ",", std::get<1>(offset),
                      ") recv_offset=(", std::get<0>(peer_offset), ",",
                      std::get<1>(peer_offset), ") send_shape=",
                      layout->send_bufs[bid][n].sizes(), " recv_shape=",
                      peer->recv_bufs[peer_bid][n].sizes());
          peer->recv_bufs[peer_bid][n]
              .view({-1})
              .copy_(layout->send_bufs[bid][n].reshape({-1}));
        }
      }
    }
  }
}

void LayoutImpl::serialize(MeshBlockImpl const* pmb, Variables& vars,
                           SyncOptions const& opts) {
  if (options->verbose()) {
    SINFO(Layout) << "serializing data into send buffers\n";
  }

  // Get my logical location
  auto iloc = loc_of(options->rank());

  // Iterate over all 2D neighbor directions
  int dy_min = opts.dy_min();
  int dy_max = opts.dy_max();
  int dx_min = opts.dx_min();
  int dx_max = opts.dx_max();

  for (int dy = dy_min; dy <= dy_max; ++dy)
    for (int dx = dx_min; dx <= dx_max; ++dx) {
      // Skip the center (self)
      if (dy == 0 && dx == 0) continue;
      if (opts.skip_corner() && std::abs(dy) + std::abs(dx) == 2) continue;
      if (pmb->options->is_physical_boundary(dy, dx, 0)) continue;

      std::tuple<int, int, int> offset(dy, dx, 0);
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

  comm->sync_device();
}

void LayoutImpl::forward(MeshBlockImpl const* pmb, Variables& vars,
                         SyncOptions const& opts,
                         std::vector<c10::intrusive_ptr<c10d::Work>>& works) {
  TORCH_CHECK(!options->no_backend(), "[Layout:forward] backend is disabled");
  TORCH_CHECK(pmb != nullptr, "[Layout:forward] MeshBlock pointer is null");

  serialize(pmb, vars, opts);
  launch_exchange(pmb, opts, works);
}

void LayoutImpl::launch_exchange(
    MeshBlockImpl const* pmb, SyncOptions const& opts,
    std::vector<c10::intrusive_ptr<c10d::Work>>& works) {
  _prepare_local_exchange(pmb, opts);
  exchange_remote(pmb, opts, works);
}

void LayoutImpl::exchange_remote(
    MeshBlockImpl const* pmb, SyncOptions const& opts,
    std::vector<c10::intrusive_ptr<c10d::Work>>& works) {
  TORCH_CHECK(!options->no_backend(),
              "[Layout:exchange_remote] backend is disabled");
  TORCH_CHECK(pmb != nullptr, "[Layout:exchange_remote] MeshBlock pointer is null");

  if (options->verbose()) {
    SINFO(Layout) << "performing communication\n";
  }

  // Get my rank
  auto rank = options->rank();

  // Get my logical location
  auto iloc = loc_of(rank);

  int dy_min = opts.dy_min();
  int dy_max = opts.dy_max();
  int dx_min = opts.dx_min();
  int dx_max = opts.dx_max();
  int dx_sgn = 1;
  int dy_sgn = 1;

  // swap the order of first block for periodic condition
  if (options->periodic_x() && options->px() == 2 && std::get<0>(iloc) == 0) {
    dx_sgn = -1;
  }

  if (options->periodic_y() && options->py() == 2 && std::get<1>(iloc) == 0) {
    dy_sgn = -1;
  }

  std::lock_guard<std::mutex> lock(g_process_comm_mutex);
  comm->group_start();

  for (int dy_ = dy_min; dy_ <= dy_max; ++dy_)
    for (int dx_ = dx_min; dx_ <= dx_max; ++dx_) {
      int dy = dy_sgn * dy_;
      int dx = dx_sgn * dx_;

      // skip the center (self)
      if (dy == 0 && dx == 0) continue;
      if (opts.skip_corner() && std::abs(dy) + std::abs(dx) == 2) continue;
      if (pmb->options->is_physical_boundary(dy, dx, 0)) continue;

      std::tuple<int, int, int> offset(dy, dx, 0);
      int nb = neighbor_rank(iloc, offset);
      if (nb < 0) continue;  // no neighbor

      int r = get_buffer_id(offset);
      int remote_process = options->owner_process_rank(nb);
      bool is_remote = remote_process != options->process_rank();

      if (is_remote) {
        int remote_local_block = options->local_block_index(nb);
        int local_block = options->local_block_index(rank);
        int send_id =
            make_comm_tag(remote_local_block, std::tuple<int, int, int>(-dy, -dx, 0),
                          opts.phyid());
        int recv_id = make_comm_tag(local_block, offset, opts.phyid());

        auto send_work = comm->pg->send(send_bufs[r], remote_process, send_id);
        works.push_back(send_work);

        auto recv_work = comm->pg->recv(recv_bufs[r], remote_process, recv_id);
        works.push_back(recv_work);
      } else if (nb == rank) {  // self-send
        int r1 = get_buffer_id(std::tuple<int, int, int>(-dy, -dx, 0));
        for (int n = 0; n < recv_bufs[r].size(); ++n)
          recv_bufs[r1][n].copy_(send_bufs[r][n]);
      }
    }

  comm->group_end();
}

void LayoutImpl::deserialize(MeshBlockImpl const* pmb, Variables& vars,
                             SyncOptions const& opts) const {
  if (options->verbose()) {
    SINFO(Layout) << "deserializing data from receive buffers\n";
  }

  comm->sync_device();

  // Get my logical location
  auto iloc = loc_of(options->rank());

  int dy_min = opts.dy_min();
  int dy_max = opts.dy_max();
  int dx_min = opts.dx_min();
  int dx_max = opts.dx_max();

  // Iterate over all 2D neighbor directions
  for (int dy = dy_min; dy <= dy_max; ++dy)
    for (int dx = dx_min; dx <= dx_max; ++dx) {
      // Skip the center (self)
      if (dy == 0 && dx == 0) continue;
      if (opts.skip_corner() && std::abs(dy) + std::abs(dx) == 2) continue;
      if (pmb->options->is_physical_boundary(dy, dx, 0)) continue;

      std::tuple<int, int, int> offset(dy, dx, 0);
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
                          std::vector<c10::intrusive_ptr<c10d::Work>>& works) {
  // Wait for all operations to complete
  for (auto& work : works) work->wait();

  // Deserialize received data into ghost zones
  deserialize(pmb, vars, opts);

  // Fill corners
  if (opts.skip_corner() && !opts.cross_panel_only()) {
    fill_corners(pmb, vars);
  }

  /*c10d::BarrierOptions op;
  op.device_ids = {options->local_rank()};
  pg->barrier(op)->wait();*/
  {
    std::lock_guard<std::mutex> lock(g_process_comm_mutex);
    comm->pg->barrier()->wait();
  }

  works.clear();
}

void LayoutImpl::_init_process_group() {
  if (options->no_backend()) return;
  comm = ProcessGroupContext::create(options);
}

#ifdef NOT_USE_C10D_NCCL
void LayoutImpl::_init_process_group() {}
#endif

}  // namespace snap
