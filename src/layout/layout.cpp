// yaml
#include <yaml-cpp/yaml.h>

// base
#include <configure.h>  // gloo and nccl

// snap
#include <snap/mesh/meshblock.hpp>
#include <snap/utils/log.hpp>

#include "cubed_sphere_layout.hpp"
#include "layout.hpp"

namespace snap {

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

  return pl;
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

        auto send_work = pg->send(send_bufs[r], remote_process, send_id);
        works.push_back(send_work);

        auto recv_work = pg->recv(recv_bufs[r], remote_process, recv_id);
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
  pg->barrier()->wait();

  works.clear();
}

void LayoutImpl::_init_process_group() {
  if (options->no_backend()) return;
  comm = ProcessGroupContext::create(options);
  pg = comm->pg;
}

#ifdef NOT_USE_C10D_NCCL
void LayoutImpl::_init_process_group() {}
#endif

}  // namespace snap
