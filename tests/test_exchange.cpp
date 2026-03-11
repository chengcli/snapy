// torch
#include <c10/cuda/CUDAFunctions.h>

// snap
#include <snap/coord/cubed_sphere_utils.hpp>
#include <snap/coord/spherical_utils.hpp>
#include <snap/mesh/mesh.hpp>

using namespace snap;

namespace {

void set_zonal_velocity(MeshBlock block, torch::Tensor const& hydro_w) {
  auto pcoord = block->pcoord;
  auto layout = block->get_layout();
  auto [rx, ry, face_id] = layout->loc_of(layout->options->rank());
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

  hydro_w[IVY] = lat.cos();
  sph_contra_to_cart_(hydro_w.narrow(0, IVX, 3), M_PI / 2. - lat, lon);
  cs_cart_to_contra_(hydro_w.narrow(0, IVX, 3), alpha, beta, face_id);
}

torch::Device select_device(LayoutOptions const& layout) {
  auto device = torch::Device(torch::kCPU);
  if (layout->backend() == "nccl") {
    TORCH_CHECK(torch::cuda::is_available(),
                "CUDA is required for backend=nccl");
    int device_index = layout->device_id();
    if (device_index < 0) device_index = layout->local_rank();
    c10::cuda::set_device(device_index);
    device = torch::Device(torch::kCUDA, device_index);
  }
  return device;
}

void seed_hydro_state(MeshBlock block, Variables& vars, torch::Device device) {
  auto pcoord = block->pcoord;
  int nc1 = pcoord->options->nc1();
  int nc2 = pcoord->options->nc2();
  int nc3 = pcoord->options->nc3();
  int rank = block->options->layout()->rank();

  auto hydro_w = torch::zeros(
      {5, nc3, nc2, nc1},
      torch::TensorOptions().dtype(torch::kFloat64).device(device));

  auto interior = block->part({0, 0, 0}, PartOptions().exterior(false));
  auto left = block->part({-1, 0, 0}, PartOptions().exterior(false));
  auto right = block->part({1, 0, 0}, PartOptions().exterior(false));
  auto bot = block->part({0, -1, 0}, PartOptions().exterior(false));
  auto top = block->part({0, 1, 0}, PartOptions().exterior(false));

  hydro_w.index(interior)[IDN] = rank + 1.0;
  hydro_w.index(interior)[IPR] = rank + 1.0;
  hydro_w.index(left)[IDN] += 0.1 * 1;
  hydro_w.index(right)[IDN] += 0.1 * 2;
  hydro_w.index(bot)[IDN] += 0.1 * 3;
  hydro_w.index(top)[IDN] += 0.1 * 6;

  auto wleft = hydro_w.index(left)[IDN];
  for (int k = 0; k < wleft.size(0); ++k)
    for (int j = 0; j < wleft.size(1); ++j)
      for (int i = 0; i < wleft.size(2); ++i) {
        wleft.index({k, j, i}) += 0.01 * j;
        wleft.index({k, j, i}) += 0.001 * k;
      }

  auto wright = hydro_w.index(right)[IDN];
  for (int k = 0; k < wright.size(0); ++k)
    for (int j = 0; j < wright.size(1); ++j)
      for (int i = 0; i < wright.size(2); ++i) {
        wright.index({k, j, i}) += 0.01 * (wright.size(1) - 1 - j);
        wright.index({k, j, i}) += 0.001 * (wright.size(0) - 1 - k);
      }

  auto wbot = hydro_w.index(bot)[IDN];
  for (int k = 0; k < wbot.size(0); ++k)
    for (int j = 0; j < wbot.size(1); ++j)
      for (int i = 0; i < wbot.size(2); ++i) {
        wbot.index({k, j, i}) += 0.01 * k;
        wbot.index({k, j, i}) += 0.001 * j;
      }

  auto wtop = hydro_w.index(top)[IDN];
  for (int k = 0; k < wtop.size(0); ++k)
    for (int j = 0; j < wtop.size(1); ++j)
      for (int i = 0; i < wtop.size(2); ++i) {
        wtop.index({k, j, i}) += 0.01 * (wtop.size(0) - 1 - k);
        wtop.index({k, j, i}) += 0.001 * (wtop.size(1) - 1 - j);
      }

  set_zonal_velocity(block, hydro_w);
  vars["hydro_w"] = hydro_w;
}

bool ghosts_changed(MeshBlock block, torch::Tensor const& before,
                    torch::Tensor const& after) {
  for (auto offset : {std::tuple<int, int, int>{-1, 0, 0},
                      std::tuple<int, int, int>{1, 0, 0},
                      std::tuple<int, int, int>{0, -1, 0},
                      std::tuple<int, int, int>{0, 1, 0}}) {
    auto ghost = block->part(offset, PartOptions().exterior(true));
    if (!torch::allclose(before.index(ghost), after.index(ghost))) {
      return true;
    }
  }
  return false;
}

}  // namespace

int main(int argc, char** argv) {
  auto block_opts = MeshBlockOptionsImpl::from_yaml("test_exchange.yaml");
  auto mesh_opts = MeshOptionsImpl::create();
  mesh_opts->block(block_opts);
  mesh_opts->blocks_per_process(3);

  auto device = select_device(block_opts->layout());
  auto mesh = Mesh(mesh_opts);
  if (device.is_cuda()) {
    mesh->to(device);
  }

  MeshVariables vars(mesh->blocks.size());
  std::vector<torch::Tensor> before(mesh->blocks.size());
  bool saw_local_neighbor = false;
  bool saw_remote_neighbor = false;

  for (int i = 0; i < mesh->blocks.size(); ++i) {
    seed_hydro_state(mesh->blocks[i], vars[i], device);
    before[i] = vars[i]["hydro_w"].clone();
  }

  mesh->initialize(vars);
  mesh->blocks.front()->get_layout()->pg->barrier()->wait();

  bool ok = true;
  for (int i = 0; i < mesh->blocks.size(); ++i) {
    auto block = mesh->blocks[i];
    auto layout = block->get_layout();
    auto iloc = layout->loc_of(layout->options->rank());

    ok = ok && torch::isfinite(vars[i]["hydro_w"]).all().item<bool>();
    ok = ok && ghosts_changed(block, before[i], vars[i]["hydro_w"]);

    for (auto offset : {std::tuple<int, int, int>{-1, 0, 0},
                        std::tuple<int, int, int>{1, 0, 0},
                        std::tuple<int, int, int>{0, -1, 0},
                        std::tuple<int, int, int>{0, 1, 0}}) {
      int nb = layout->neighbor_rank(iloc, offset);
      if (nb < 0) continue;
      if (layout->options->owner_process_rank(nb) ==
          layout->options->process_rank()) {
        saw_local_neighbor = true;
      } else {
        saw_remote_neighbor = true;
      }
    }
  }

  ok = ok && saw_local_neighbor && saw_remote_neighbor;
  mesh->blocks.front()->get_layout()->pg->barrier()->wait();

  if (!ok) {
    std::cerr << "legacy cubed-sphere exchange regression failed on process "
              << block_opts->layout()->process_rank() << std::endl;
    return 1;
  }

  return 0;
}
