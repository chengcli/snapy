// C/C++
#include <future>

// snap
#include <snap/mesh/mesh.hpp>

namespace snap {

namespace {
MeshBlockOptions clone_block_options(MeshBlockOptions const& src) {
  auto dst = std::make_shared<MeshBlockOptionsImpl>(*src);
  auto layout = std::make_shared<LayoutOptionsImpl>(*src->layout());
  dst->layout(layout);
  return dst;
}
}  // namespace

MeshImpl::MeshImpl(MeshOptions const& options_) : options(options_) { reset(); }

void MeshImpl::reset() {
  blocks.clear();
  TORCH_CHECK(options->block() != nullptr, "Mesh requires MeshBlockOptions");

  auto base = options->block();
  auto base_layout = base->layout();
  TORCH_CHECK(base_layout != nullptr, "Mesh requires LayoutOptions");
  TORCH_CHECK(options->blocks_per_process() > 0,
              "blocks_per_process must be positive");

  base_layout->blocks_per_process(options->blocks_per_process());
  base_layout->world_size(base_layout->process_world_size() *
                          options->blocks_per_process());

  for (int i = 0; i < options->blocks_per_process(); ++i) {
    auto block_opts = clone_block_options(base);
    auto layout = block_opts->layout();
    layout->blocks_per_process(options->blocks_per_process());
    layout->world_size(layout->process_world_size() *
                       options->blocks_per_process());
    layout->rank(layout->global_block_rank(layout->process_rank(), i));
    auto block = register_module("block" + std::to_string(i), MeshBlock(block_opts));
    blocks.push_back(block);
  }
}

void MeshImpl::initialize(MeshVariables& vars,
                          std::vector<char const*> const& restart_files) {
  TORCH_CHECK(vars.size() == blocks.size(),
              "Mesh::initialize expects one Variables map per local MeshBlock");
  for (int i = 0; i < blocks.size(); ++i) {
    char const* restart = nullptr;
    if (i < restart_files.size()) restart = restart_files[i];
    blocks[i]->initialize(vars[i], restart);
  }

  SyncOptions prim_opts;
  prim_opts.interpolate(true).type(kPrimitive);
  _exchange_all(vars, prim_opts, "hydro_w");

  if (blocks.front()->pscalar->nvar() > 0 && vars.front().count("scalar_r")) {
    SyncOptions scalar_opts;
    scalar_opts.interpolate(true).type(kScalar);
    _exchange_all(vars, scalar_opts, "scalar_r");
  }

  for (int i = 0; i < blocks.size(); ++i) {
    vars[i]["hydro_u"] = blocks[i]->phydro->peos->compute("W->U", {vars[i]["hydro_w"]});
    if (blocks[i]->pscalar->nvar() > 0 && vars[i].count("scalar_r")) {
      vars[i]["scalar_s"] = vars[i]["hydro_w"][IDN] * vars[i]["scalar_r"];
    }
  }
}

double MeshImpl::max_time_step(MeshVariables const& vars) {
  TORCH_CHECK(vars.size() == blocks.size(),
              "Mesh::max_time_step expects one Variables map per local MeshBlock");

  double dt_local = 1.e99;
  for (int i = 0; i < blocks.size(); ++i) {
    dt_local = std::min(dt_local, blocks[i]->local_max_time_step(vars[i]));
  }

  auto device = blocks.front()->device();
  auto dt_tensor =
      torch::tensor({dt_local}, torch::dtype(torch::kFloat64).device(device));
  std::vector<at::Tensor> dt_reduce = {dt_tensor};
  c10d::AllreduceOptions op;
  op.reduceOp = c10d::ReduceOp::MIN;
  blocks.front()->get_layout()->pg->allreduce(dt_reduce, op)->wait();

  auto dt = dt_reduce[0].item<double>();
  auto redo = blocks.front()->pintg->current_redo;
  auto cfl = blocks.front()->pintg->options->cfl();
  return pow(2., -redo) * cfl * dt;
}

void MeshImpl::forward(MeshVariables& vars, double dt, int stage) {
  TORCH_CHECK(vars.size() == blocks.size(),
              "Mesh::forward expects one Variables map per local MeshBlock");

  std::vector<std::future<void>> jobs;
  jobs.reserve(blocks.size());
  for (int i = 0; i < blocks.size(); ++i) {
    jobs.push_back(std::async(std::launch::async, [&, i]() {
      blocks[i]->advance_local(vars[i], dt, stage);
    }));
  }
  for (auto& job : jobs) {
    job.get();
  }

  SyncOptions cons_opts;
  cons_opts.interpolate(true).type(kConserved);
  _exchange_all(vars, cons_opts, "hydro_u");

  if (blocks.front()->pscalar->nvar() > 0) {
    SyncOptions scalar_opts;
    scalar_opts.interpolate(true).type(kScalar);
    _exchange_all(vars, scalar_opts, "scalar_s");
  }
}

void MeshImpl::_exchange_all(MeshVariables& vars, SyncOptions const& opts,
                             char const* var_name) {
  std::vector<c10::intrusive_ptr<c10d::Work>> works;

  for (int i = 0; i < blocks.size(); ++i) {
    Variables sync_vars;
    sync_vars[var_name] = vars[i].at(var_name);
    blocks[i]->get_layout()->serialize(blocks[i].get(), sync_vars, opts);
  }

  for (int i = 0; i < blocks.size(); ++i) {
    auto block = blocks[i];
    auto layout = block->get_layout();
    auto rank = layout->options->rank();
    auto iloc = layout->loc_of(rank);
    bool has_remote_neighbor = false;

    int dy_min = opts.dy_min();
    int dy_max = opts.dy_max();
    int dx_min = opts.dx_min();
    int dx_max = opts.dx_max();

    for (int dy = dy_min; dy <= dy_max; ++dy) {
      for (int dx = dx_min; dx <= dx_max; ++dx) {
        if (dy == 0 && dx == 0) continue;
        if (opts.skip_corner() && std::abs(dy) + std::abs(dx) == 2) continue;
        if (block->options->is_physical_boundary(dy, dx, 0)) continue;

        std::tuple<int, int, int> offset(dy, dx, 0);
        int nb = layout->neighbor_rank(iloc, offset);
        if (nb < 0) continue;

        int bid = get_buffer_id(offset);
        int local_process = layout->options->process_rank();
        int neighbor_process = layout->options->owner_process_rank(nb);
        if (neighbor_process == local_process) {
          int neighbor_local = layout->options->local_block_index(nb);
          auto peer = blocks[neighbor_local]->get_layout();
          int peer_bid =
              get_buffer_id(std::tuple<int, int, int>(-dy, -dx, 0));
          for (int n = 0; n < layout->send_bufs[bid].size(); ++n) {
            peer->recv_bufs[peer_bid][n].copy_(layout->send_bufs[bid][n]);
          }
        } else {
          has_remote_neighbor = true;
        }
      }
    }

    if (has_remote_neighbor) {
      layout->exchange_remote(block.get(), opts, works);
    }
  }

  for (int i = 0; i < blocks.size(); ++i) {
    Variables sync_vars;
    sync_vars[var_name] = vars[i].at(var_name);
    blocks[i]->get_layout()->finalize(blocks[i].get(), sync_vars, opts, works);
  }
}

}  // namespace snap
