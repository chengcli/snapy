// C/C++
#include <algorithm>
#include <future>
#include <iomanip>
#include <iostream>
#include <limits>

// snap
#include <snap/mesh/mesh.hpp>
#include <snap/utils/log.hpp>
#include <snap/utils/signal_handler.hpp>

namespace snap {

namespace {
MeshBlockOptions clone_block_options(MeshBlockOptions const& src) {
  auto dst = std::make_shared<MeshBlockOptionsImpl>(*src);
  auto layout = std::make_shared<LayoutOptionsImpl>(*src->layout());
  dst->layout(layout);
  if (src->coord() != nullptr) {
    dst->coord(src->coord()->clone());
  }
  return dst;
}

std::tuple<int, int, int> remap_exchange_offset(Layout const& layout,
                                                std::tuple<int, int, int> iloc,
                                                int dy, int dx) {
  int dx_sgn = 1;
  int dy_sgn = 1;

  if (layout->options->periodic_x() && layout->options->px() == 2 &&
      std::get<0>(iloc) == 0) {
    dx_sgn = -1;
  }

  if (layout->options->periodic_y() && layout->options->py() == 2 &&
      std::get<1>(iloc) == 0) {
    dy_sgn = -1;
  }

  return std::tuple<int, int, int>(dy_sgn * dy, dx_sgn * dx, 0);
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
    if (block_opts->coord() != nullptr) {
      block_opts->coord()->repartition(layout);
    }
    auto block = register_module("block" + std::to_string(i), MeshBlock(block_opts));
    blocks.push_back(block);
  }
}

double MeshImpl::initialize(MeshVariables& vars,
                            std::vector<char const*> const& restart_files) {
  TORCH_CHECK(vars.size() == blocks.size(),
              "Mesh::initialize expects one Variables map per local MeshBlock");

  bool has_restart = false;
  for (int i = 0; i < blocks.size(); ++i) {
    if (i < restart_files.size() && restart_files[i] != nullptr) {
      has_restart = true;
      break;
    }
  }

  if (has_restart) {
    double current_time = 0.;
    for (int i = 0; i < blocks.size(); ++i) {
      char const* restart = nullptr;
      if (i < restart_files.size()) restart = restart_files[i];
      auto block_time = blocks[i]->initialize(vars[i], restart);
      if (i == 0) {
        current_time = block_time;
      } else {
        TORCH_CHECK(block_time == current_time,
                    "Mesh::initialize requires identical restart times across "
                    "local MeshBlocks, expected ",
                    current_time, " but got ", block_time, " on block ", i);
      }
    }
    return current_time;
  }

  auto pg = blocks.front()->get_layout()->pg;
  pg->barrier()->wait();
  SignalHandler::GetInstance();

  for (int i = 0; i < blocks.size(); ++i) {
    blocks[i]->initialize_local(vars[i]);
  }

  SyncOptions prim_opts;
  prim_opts.interpolate(true).type(kPrimitive);
  exchange(vars, prim_opts, "hydro_w");

  if (blocks.front()->pscalar->nvar() > 0 && vars.front().count("scalar_r")) {
    SyncOptions scalar_opts;
    scalar_opts.interpolate(true).type(kScalar);
    exchange(vars, scalar_opts, "scalar_r");
  }

  for (int i = 0; i < blocks.size(); ++i) {
    blocks[i]->finalize_initialization(vars[i]);
  }

  return 0.;
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
  exchange(vars, cons_opts, "hydro_u");

  if (blocks.front()->pscalar->nvar() > 0) {
    SyncOptions scalar_opts;
    scalar_opts.interpolate(true).type(kScalar);
    exchange(vars, scalar_opts, "scalar_s");
  }
}

void MeshImpl::exchange(MeshVariables& vars, SyncOptions const& opts,
                        char const* var_name) {
  _exchange_all(vars, opts, var_name);
}

void MeshImpl::make_outputs(MeshVariables const& vars, double current_time,
                            bool final_write) {
  TORCH_CHECK(vars.size() == blocks.size(),
              "Mesh::make_outputs expects one Variables map per local MeshBlock");
  for (int i = 0; i < blocks.size(); ++i) {
    blocks[i]->make_outputs(vars[i], current_time, final_write);
  }
}

void MeshImpl::print_cycle_info(MeshVariables const& vars, double time,
                                double dt) const {
  TORCH_CHECK(vars.size() == blocks.size(),
              "Mesh::print_cycle_info expects one Variables map per local "
              "MeshBlock");

  auto root = blocks.front();
  auto pintg = root->pintg;
  if (pintg->options->ncycle_out() == 0 ||
      root->cycle % pintg->options->ncycle_out() != 0) {
    return;
  }

  const int dt_precision = std::numeric_limits<double>::max_digits10 - 3;
  bool compute_mass = false;
  bool compute_energy = false;

  if (vars.front().count("hydro_u")) {
    compute_mass = true;
    compute_energy = root->phydro->peos->nvar() > IPR;
  }

  SINFO() << "cycle=" << root->cycle << " redo=" << pintg->current_redo
          << std::scientific << std::setprecision(dt_precision)
          << " time=" << time << " dt=" << dt;

  c10d::ReduceOptions opsum;
  opsum.reduceOp = c10d::ReduceOp::SUM;
  opsum.rootRank = root->options->layout()->process_root_rank();

  torch::Tensor local_sum;
  if (compute_mass || compute_energy) {
    for (int i = 0; i < blocks.size(); ++i) {
      auto interior = blocks[i]->part({0, 0, 0}, PartOptions().exterior(false));
      auto vol = blocks[i]->pcoord->cell_volume();
      auto hydro_u_tot = vars[i].at("hydro_u") * vol;
      auto block_sum = hydro_u_tot.index(interior).sum({1, 2, 3});
      if (!local_sum.defined()) {
        local_sum = block_sum.clone();
      } else {
        local_sum += block_sum;
      }
    }
  }

  if (local_sum.defined()) {
    std::vector<at::Tensor> sum = {local_sum};
    root->get_layout()->pg->reduce(sum, opsum)->wait();

    if (compute_mass) {
      auto mass = sum[0][IDN];
      SINFO() << std::scientific << std::setprecision(dt_precision)
              << " mass0=" << mass.item<double>();

      int ny = local_sum.size(0) - 5;
      if (ny > 0) {
        for (int n = 0; n < ny; ++n) {
          mass += sum[0][ICY + n];
        }
        SINFO() << std::scientific << std::setprecision(dt_precision)
                << " masst=" << mass.item<double>();
      }
    }

    if (compute_energy) {
      SINFO() << std::scientific << std::setprecision(dt_precision)
              << " energy=" << sum[0][IPR].item<double>();
    }
  }

  SINFO() << std::endl;
}

int MeshImpl::check_redo(MeshVariables& vars) {
  TORCH_CHECK(vars.size() == blocks.size(),
              "Mesh::check_redo expects one Variables map per local MeshBlock");

  int redo = 0;
  for (int i = 0; i < blocks.size(); ++i) {
    int err = blocks[i]->check_redo(vars[i]);
    if (err < 0) return -1;
    redo = std::max(redo, err);
  }
  return redo;
}

void MeshImpl::set_cycle(int cycle) {
  for (auto& block : blocks) {
    block->cycle = cycle;
  }
}

void MeshImpl::finalize(MeshVariables const& vars, double time) {
  TORCH_CHECK(vars.size() == blocks.size(),
              "Mesh::finalize expects one Variables map per local MeshBlock");

  if (blocks.size() == 1) {
    blocks.front()->finalize(vars.front(), time);
    return;
  }

  make_outputs(vars, time, /*final_write=*/true);

  auto root = blocks.front();
  auto sig = SignalHandler::GetInstance();
  if (sig->GetSignalFlag(SIGTERM) != 0) {
    std::cout << std::endl << "Terminating on Terminate signal" << std::endl;
  } else if (sig->GetSignalFlag(SIGINT) != 0) {
    std::cout << std::endl << "Terminating on Interrupt signal" << std::endl;
  } else if (sig->GetSignalFlag(SIGALRM) != 0) {
    std::cout << std::endl << "Terminating on wall-time limit" << std::endl;
  } else if (root->pintg->options->nlim() >= 0 &&
             root->cycle >= root->pintg->options->nlim()) {
    std::cout << std::endl << "Terminating on cycle limit" << std::endl;
  } else if (time >= root->pintg->options->tlim()) {
    std::cout << std::endl << "Terminating on time limit" << std::endl;
  } else {
    std::cout << std::endl << "Terminating abnormally" << std::endl;
  }

  std::cout << "time=" << time << " cycle=" << root->cycle - 1 << std::endl;
  std::cout << "tlim=" << root->pintg->options->tlim()
            << " nlim=" << root->pintg->options->nlim() << std::endl;

  for (auto& block : blocks) {
    auto layout = block->get_layout();
    layout->send_bufs.clear();
    layout->send_bufs.shrink_to_fit();
    layout->recv_bufs.clear();
    layout->recv_bufs.shrink_to_fit();
  }

  root->get_layout()->pg->barrier()->wait();
  root->get_layout()->pg->shutdown();
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

    for (int dy_ = dy_min; dy_ <= dy_max; ++dy_) {
      for (int dx_ = dx_min; dx_ <= dx_max; ++dx_) {
        if (dy_ == 0 && dx_ == 0) continue;
        if (opts.skip_corner() && std::abs(dy_) + std::abs(dx_) == 2) continue;
        if (block->options->layout()->type() != "cubed-sphere" &&
            block->options->is_physical_boundary(dy_, dx_, 0)) {
          continue;
        }

        auto offset = remap_exchange_offset(layout, iloc, dy_, dx_);
        auto [dy, dx, dz] = offset;
        (void)dz;
        int nb = layout->neighbor_rank(iloc, offset);
        if (nb < 0) continue;

        int bid = get_buffer_id(offset);
        int local_process = layout->options->process_rank();
        int neighbor_process = layout->options->owner_process_rank(nb);
        if (neighbor_process == local_process) {
          int neighbor_local = layout->options->local_block_index(nb);
          auto peer = blocks[neighbor_local]->get_layout();
          auto peer_offset = std::tuple<int, int, int>(-dy, -dx, 0);
          int peer_bid = get_buffer_id(peer_offset);
          for (int n = 0; n < layout->send_bufs[bid].size(); ++n) {
            TORCH_CHECK(
                peer->recv_bufs[peer_bid][n].numel() ==
                    layout->send_bufs[bid][n].numel(),
                "local exchange size mismatch from rank ", rank, " to rank ",
                nb, " send_offset=(", dy, ",", dx, ") recv_offset=(",
                std::get<0>(peer_offset), ",", std::get<1>(peer_offset),
                ") send_shape=", layout->send_bufs[bid][n].sizes(),
                " recv_shape=", peer->recv_bufs[peer_bid][n].sizes());
            peer->recv_bufs[peer_bid][n]
                .view({-1})
                .copy_(layout->send_bufs[bid][n].reshape({-1}));
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
