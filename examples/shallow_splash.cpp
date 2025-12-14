// yaml
#include <yaml-cpp/yaml.h>

// snap
#include <snap/coord/cubed_sphere_utils.hpp>
#include <snap/mesh/meshblock.hpp>

using namespace snap;

int main(int argc, char** argv) {
  torch::set_num_threads(1);
  torch::set_num_interop_threads(1);

  auto config = YAML::LoadFile("shallow_splash.yaml");

  auto phi = config["problem"]["phi"].as<double>();
  auto dphi = config["problem"]["dphi"].as<double>();
  auto radius = config["problem"]["radius"].as<double>();

  auto block_op = MeshBlockOptionsImpl::from_yaml("shallow_splash.yaml");
  auto block = MeshBlock(block_op);
  auto device = torch::kCPU;
  if (torch::cuda::is_available()) {
    std::cout << "Running on CUDA" << std::endl;
    device = torch::kCUDA;
  }

  block->to(device);

  // initial conditions
  auto pcoord = block->phydro->pcoord;
  auto peos = block->phydro->peos;

  // coordinates
  int r = get_rank();
  auto [rx, ry, face_id] = block->get_layout()->loc_of(r);
  auto face = CS_FACE_NAMES[face_id];

  auto mesh = torch::meshgrid({pcoord->x3v, pcoord->x2v, pcoord->x1v}, "ij");
  auto alpha = mesh[1];
  auto beta = mesh[0];
  auto r_planet = mesh[2];
  auto [lon, lat] = cs_ab_to_lonlat(face, alpha, beta);

  int nc1 = pcoord->options->nc1();
  int nc2 = pcoord->options->nc2();
  int nc3 = pcoord->options->nc3();
  int nvar = peos->nvar();

  auto w = torch::zeros(
      {nvar, nc3, nc2, nc1},
      torch::TensorOptions().dtype(torch::kFloat64).device(device));

  auto dist = (M_PI / 2. - lat) * r_planet;

  // w[IDN] = torch::where(torch::logical_and(dist < radius, lat > M_PI / 4.),
  //                       phi + dphi, phi);
  w[IDN] = phi;
  w[IVX] = 0.;
  w[IVY] = 0.;

  // initialize
  std::map<std::string, torch::Tensor> vars;
  vars["hydro_w"] = w;
  block->initialize(vars);

  double current_time = 0.;
  block->make_outputs(vars, current_time);

  while (!block->pintg->stop(block->cycle++, current_time)) {
    auto dt = block->max_time_step(vars);
    block->print_cycle_info(vars, current_time, dt);

    // main loop
    for (int stage = 0; stage < block->pintg->stages.size(); ++stage) {
      block->forward(vars, dt, stage);
    }

    int err = block->check_redo(vars);
    if (err > 0) continue;  // redo this step with smaller dt
    if (err < 0) break;     // terminate simulation

    // make outputs
    current_time += dt;
    block->make_outputs(vars, current_time);
  }

  block->finalize(vars, current_time);
}
