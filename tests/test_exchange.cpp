// C/C++
#include <cstdio>

// torch
#include <c10/cuda/CUDAFunctions.h>

// yaml
#include <yaml-cpp/yaml.h>

// snap
#include <snap/coord/cubed_sphere_utils.hpp>
#include <snap/coord/spherical_utils.hpp>
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
  auto device = hydro_w.device();
  alpha = alpha.to(device);
  beta = beta.to(device);
  lon = lon.to(device);
  lat = lat.to(device);

  // hydro_w[IVZ] = lat.cos();
  hydro_w[IVY] = lat.cos();

  sph_contra_to_cart_(hydro_w.narrow(0, IVX, 3), M_PI / 2. - lat, lon);
  cs_cart_to_contra_(hydro_w.narrow(0, IVX, 3), alpha, beta, face_id);
}

int main(int argc, char** argv) {
  auto op = MeshBlockOptionsImpl::from_yaml("test_exchange.yaml");
  auto block = MeshBlock(op);

  auto device = torch::Device(torch::kCPU);
  if (op->layout()->backend() == "nccl") {
    TORCH_CHECK(torch::cuda::is_available(),
                "CUDA is required for backend=nccl");
    int device_index = op->layout()->device_id();
    if (device_index < 0) device_index = op->layout()->local_rank();
    c10::cuda::set_device(device_index);
    device = torch::Device(torch::kCUDA, device_index);
  }
  if (device.is_cuda()) {
    block->to(device);
  }

  auto pcoord = block->pcoord;

  int nc1 = pcoord->options->nc1();
  int nc2 = pcoord->options->nc2();
  int nc3 = pcoord->options->nc3();
  int nghost = pcoord->options->nghost();

  auto w = torch::zeros(
      {5, nc3, nc2, nc1},
      torch::TensorOptions().dtype(torch::kFloat64).device(device));
  int r = block->options->layout()->rank();

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
  block->get_layout()->pg->barrier()->wait();

  auto w_left = -get_rank() * torch::ones_like(vars["hydro_w"]);
  auto w_right = get_rank() * torch::ones_like(vars["hydro_w"]);
  auto w_left_before = w_left.clone();
  auto w_right_before = w_right.clone();

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

  bool ok = true;
  ok = ok && torch::isfinite(vars["hydro_w"]).all().item<bool>();
  ok = ok && torch::isfinite(w_left).all().item<bool>();
  ok = ok && torch::isfinite(w_right).all().item<bool>();
  ok = ok && !torch::allclose(w_left, w_left_before);
  ok = ok && !torch::allclose(w_right, w_right_before);
  block->get_layout()->pg->barrier()->wait();

  if (!ok) {
    std::cerr << "legacy cubed-sphere exchange regression failed on rank " << r
              << std::endl;
    return 1;
  }

  return 0;
}
