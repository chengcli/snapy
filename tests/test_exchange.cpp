// C/C++
#include <cstdio>

// yaml
#include <yaml-cpp/yaml.h>

// base
#include <configure.h>

// torch
#include <torch/csrc/distributed/c10d/ProcessGroupGloo.hpp>
#include <torch/csrc/distributed/c10d/TCPStore.hpp>

// snap
#include <snap/coord/cubed_sphere_utils.hpp>
#include <snap/coord/spherical_utils.hpp>
#include <snap/layout/distributed.hpp>
#include <snap/mesh/meshblock.hpp>

using namespace snap;

// u = cos(lat)
void set_zonal_velocity(MeshBlock pmb, torch::Tensor const& hydro_w) {
  auto pcoord = pmb->pcoord;
  auto playout = pmb->get_layout();

  int r = get_rank();
  auto [rx, ry, face_id] = playout->loc_of(r);
  auto face = CS_FACE_NAMES[face_id];

  auto mesh = torch::meshgrid({pcoord->x3v, pcoord->x2v, pcoord->x1v}, "ij");
  auto alpha = mesh[1];
  auto beta = mesh[0];
  auto [lon, lat] = cs_ab_to_lonlat(face, alpha, beta);

  // hydro_w[IVZ] = lat.cos();
  hydro_w[IVY] = lat.cos();

  sph_contra_to_cart_(hydro_w.narrow(0, IVX, 3), M_PI / 2. - lat, lon);
  cs_cart_to_contra_(hydro_w.narrow(0, IVX, 3), alpha, beta);
}

int main(int argc, char** argv) {
  auto op = MeshBlockOptionsImpl::from_yaml("test_exchange.yaml");

  // Initialize the distributed process group using torch.distributed C++
  // classes (mirroring what torch.distributed.init_process_group() does from
  // Python)
  if (!op->layout()->no_backend()) {
    auto layout_op = op->layout();

    // Build TCPStore
    c10d::TCPStoreOptions store_opts;
    store_opts.port = layout_op->master_port();
    store_opts.numWorkers = layout_op->world_size();
    store_opts.isServer = (layout_op->rank() == layout_op->root_rank());
    auto store = c10::make_intrusive<c10d::TCPStore>(layout_op->master_addr(),
                                                     store_opts);

    // Build ProcessGroupGloo
    auto gloo_opts = c10d::ProcessGroupGloo::Options::create();
    gloo_opts->devices.push_back(c10d::ProcessGroupGloo::createDefaultDevice());
    auto pg = c10::make_intrusive<c10d::ProcessGroupGloo>(
        store, layout_op->rank(), layout_op->world_size(), gloo_opts);

    // Register the process group with snap's distributed state
    snap::set_process_group(pg);
    snap::get_process_group()->barrier()->wait();
  }

  auto block = MeshBlock(op);

  auto device = torch::kCPU;
  // if (torch::cuda::is_available()) {
  //  std::cout << "Running on CUDA" << std::endl;
  //   device = torch::kCUDA;
  // }

  auto pcoord = block->pcoord;

  int nc1 = pcoord->options->nc1();
  int nc2 = pcoord->options->nc2();
  int nc3 = pcoord->options->nc3();
  int nghost = pcoord->options->nghost();

  auto w = torch::zeros(
      {5, nc3, nc2, nc1},
      torch::TensorOptions().dtype(torch::kFloat64).device(device));
  int r = block->options->layout()->rank();

  if (r == 0) {
    block->named_modules()["layout"]->pretty_print(std::cout);
  }

  auto [rx, ry, face] = block->get_layout()->loc_of(r);

  auto interior = block->part({0, 0, 0}, PartOptions().exterior(false));
  auto left = block->part({-1, 0, 0}, PartOptions().exterior(false));
  auto right = block->part({1, 0, 0}, PartOptions().exterior(false));
  auto bot = block->part({0, -1, 0}, PartOptions().exterior(false));
  auto top = block->part({0, 1, 0}, PartOptions().exterior(false));

  w.index(interior)[IDN] = r + 1.0;
  w.index(interior)[IPR] = r + 1.0;

  // set up internal density as <face>.0 + <side> * 0.1
  // for cells within nghost zones
  w.index(interior)[IDN] = r + 1.0;
  w.index(left)[IDN] += 0.1 * 1;
  w.index(right)[IDN] += 0.1 * 2;
  w.index(bot)[IDN] += 0.1 * 3;
  w.index(top)[IDN] += 0.1 * 6;

  auto wleft = w.index(left)[IDN];
  for (int k = 0; k < wleft.size(0); ++k)
    for (int j = 0; j < wleft.size(1); ++j)
      for (int i = 0; i < wleft.size(2); ++i) {
        wleft.index({k, j, i}) += 0.01 * j;
        wleft.index({k, j, i}) += 0.001 * k;
      }

  auto wright = w.index(right)[IDN];
  for (int k = 0; k < wright.size(0); ++k)
    for (int j = 0; j < wright.size(1); ++j)
      for (int i = 0; i < wright.size(2); ++i) {
        wright.index({k, j, i}) += 0.01 * (wright.size(1) - 1 - j);
        wright.index({k, j, i}) += 0.001 * (wright.size(0) - 1 - k);
      }

  auto wbot = w.index(bot)[IDN];
  for (int k = 0; k < wbot.size(0); ++k)
    for (int j = 0; j < wbot.size(1); ++j)
      for (int i = 0; i < wbot.size(2); ++i) {
        wbot.index({k, j, i}) += 0.01 * k;
        wbot.index({k, j, i}) += 0.001 * j;
      }

  auto wtop = w.index(top)[IDN];
  for (int k = 0; k < wtop.size(0); ++k)
    for (int j = 0; j < wtop.size(1); ++j)
      for (int i = 0; i < wtop.size(2); ++i) {
        wtop.index({k, j, i}) += 0.01 * (wtop.size(0) - 1 - k);
        wtop.index({k, j, i}) += 0.001 * (wtop.size(1) - 1 - j);
      }

  std::map<std::string, torch::Tensor> vars;
  vars["hydro_w"] = w;
  set_zonal_velocity(block, vars["hydro_w"]);

  block->initialize(vars);
  snap::get_process_group()->barrier()->wait();

  auto w_left = -get_rank() * torch::ones_like(vars["hydro_w"]);
  auto w_right = get_rank() * torch::ones_like(vars["hydro_w"]);

  SyncOptions sync_opts;
  sync_opts.cross_panel_only(true).interpolate(false).type(kPrimitive);

  Variables send_vars;
  send_vars["hydro_wl:+"] = w_left;
  send_vars["hydro_wr:-"] = w_right;

  std::vector<c10::intrusive_ptr<c10d::Work>> works;
  auto playout = block->get_layout();

  playout->forward(block.get(), send_vars, sync_opts.dim(SyncOptions::DIM2),
                   works);
  playout->forward(block.get(), send_vars, sync_opts.dim(SyncOptions::DIM3),
                   works);

  playout->finalize(block.get(), send_vars, sync_opts.dim(SyncOptions::DIM2),
                    works);
  playout->finalize(block.get(), send_vars, sync_opts.dim(SyncOptions::DIM3),
                    works);

  for (int i = 0; i < block->options->layout()->world_size(); ++i) {
    if (i == r) {
      std::cout << fmt::format("rx = {}, ry = {}, face = {}, rank = {}", rx, ry,
                               face, r)
                << std::endl;
      std::cout << "hydro_w[IDN] = \n"
                << vars["hydro_w"][IDN].squeeze().flip(0) << std::endl;
      std::cout << "hydro_w[IVX] = \n"
                << vars["hydro_w"][IVX].squeeze().flip(0) << std::endl;
      std::cout << "hydro_w[IVY] = \n"
                << vars["hydro_w"][IVY].squeeze().flip(0) << std::endl;
      std::cout << "hydro_w[IVZ] = \n"
                << vars["hydro_w"][IVZ].squeeze().flip(0) << std::endl;

      std::cout << std::endl
                << "w_left[IDN] = \n"
                << w_left[IDN].squeeze().flip(0) << std::endl;
      std::cout << std::endl
                << "w_right[IDN] = \n"
                << w_right[IDN].squeeze().flip(0) << std::endl;
    }
    snap::get_process_group()->barrier()->wait();
  }

  block->make_outputs(vars, 0.);

  return 0;
}
