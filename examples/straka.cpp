// C/C++
#include <algorithm>
#include <string>

// yaml
#include <yaml-cpp/yaml.h>

// kintera
#include <kintera/constants.h>
#include <kintera/species.hpp>

// snap
#include <snap/snap.h>

#include <snap/mesh/mesh.hpp>

using namespace snap;

namespace {

torch::Device select_device(Mesh& mesh, MeshBlockOptions const& block_opts) {
  auto device = torch::Device(torch::kCPU);
  if (torch::cuda::is_available() && block_opts->layout()->backend() == "nccl") {
    std::cout << "Running on CUDA" << std::endl;
    device = mesh->blocks.front()->get_layout()->pg->getBoundDeviceId().value();
  }
  return device;
}

void initialize_block(MeshBlock block, Variables& vars, YAML::Node const& config,
                      torch::Device const& device) {
  auto p0 = config["problem"]["p0"].as<double>();
  auto Ts = config["problem"]["Ts"].as<double>();
  auto xc = config["problem"]["xc"].as<double>();
  auto zc = config["problem"]["zc"].as<double>();
  auto xr = config["problem"]["xr"].as<double>();
  auto zr = config["problem"]["zr"].as<double>();
  auto dT = config["problem"]["dT"].as<double>();
  auto grav = -config["forcing"]["const-gravity"]["grav1"].as<double>();

  auto pcoord = block->pcoord;
  auto peos = block->phydro->peos;

  auto Rd = kintera::constants::Rgas / peos->species_weight();
  auto cv = peos->species_cv_ref();
  auto cp = cv + Rd;

  auto grids = torch::meshgrid({pcoord->x3v, pcoord->x2v, pcoord->x1v}, "ij");
  auto x1v = grids[2];
  auto x2v = grids[1];

  int nc1 = pcoord->options->nc1();
  int nc2 = pcoord->options->nc2();
  int nc3 = pcoord->options->nc3();
  int nvar = peos->nvar();

  auto w = torch::zeros(
      {nvar, nc3, nc2, nc1},
      torch::TensorOptions().dtype(torch::kFloat64).device(device));

  auto L = torch::sqrt(((x2v - xc) / xr).square() + ((x1v - zc) / zr).square());
  auto temp = Ts - grav * x1v / cp;

  w[IPR] = p0 * torch::pow(temp / Ts, cp / Rd);
  temp += torch::where(L <= 1, dT * (torch::cos(L * M_PI) + 1.) / 2., 0);
  w[IDN] = w[IPR] / (Rd * temp);

  vars["hydro_w"] = w;
  block->user_output_callback = [Rd, cp, p0](Variables const& vars) {
    auto w = vars.at("hydro_w");
    auto temp = w[IPR] / (w[IDN] * Rd);

    Variables out;
    out["temp"] = temp;
    out["theta"] = temp * (p0 / w[IPR]).pow(Rd / cp);
    return out;
  };
}

int mesh_redo(Mesh& mesh, MeshVariables& vars) {
  int redo = 0;
  for (int i = 0; i < mesh->blocks.size(); ++i) {
    int err = mesh->blocks[i]->check_redo(vars[i]);
    if (err < 0) return -1;
    redo = std::max(redo, err);
  }
  return redo;
}

void set_cycle(Mesh& mesh, int cycle) {
  for (auto& block : mesh->blocks) {
    block->cycle = cycle;
  }
}

void make_outputs(Mesh& mesh, MeshVariables const& vars, double time) {
  for (int i = 0; i < mesh->blocks.size(); ++i) {
    mesh->blocks[i]->make_outputs(vars[i], time);
  }
}

void finalize_outputs(Mesh& mesh, MeshVariables const& vars, double time) {
  if (mesh->blocks.size() == 1) {
    mesh->blocks.front()->finalize(vars.front(), time);
    return;
  }

  for (int i = 0; i < mesh->blocks.size(); ++i) {
    mesh->blocks[i]->make_outputs(vars[i], time, /*final_write=*/true);
  }

  auto root = mesh->blocks.front();
  std::cout << std::endl << "Terminating on time limit" << std::endl;
  std::cout << "time=" << time << " cycle=" << root->cycle - 1 << std::endl;
  std::cout << "tlim=" << root->pintg->options->tlim()
            << " nlim=" << root->pintg->options->nlim() << std::endl;

  for (auto& block : mesh->blocks) {
    auto layout = block->get_layout();
    layout->send_bufs.clear();
    layout->send_bufs.shrink_to_fit();
    layout->recv_bufs.clear();
    layout->recv_bufs.shrink_to_fit();
  }

  root->get_layout()->pg->barrier()->wait();
  root->get_layout()->pg->shutdown();
}

}  // namespace

int main(int argc, char** argv) {
  torch::set_num_threads(1);
  torch::set_num_interop_threads(1);

  std::string input_file = argc > 1 ? argv[1] : "straka.yaml";
  auto config = YAML::LoadFile(input_file);

  auto op_block = MeshBlockOptionsImpl::from_yaml(input_file);
  auto op_mesh = MeshOptionsImpl::create();
  op_mesh->block(op_block);
  op_mesh->blocks_per_process(
      config["distribute"]["blocks_per_process"].as<int>(1));

  auto mesh = Mesh(op_mesh);
  auto device = select_device(mesh, op_block);
  mesh->to(device);

  MeshVariables vars(mesh->blocks.size());
  for (int i = 0; i < mesh->blocks.size(); ++i) {
    initialize_block(mesh->blocks[i], vars[i], config, device);
  }

  double current_time = 0.;
  mesh->initialize(vars);
  make_outputs(mesh, vars, current_time);

  int cycle = 0;
  while (!mesh->blocks.front()->pintg->stop(cycle, current_time)) {
    ++cycle;
    set_cycle(mesh, cycle);

    auto dt = mesh->max_time_step(vars);
    mesh->blocks.front()->print_cycle_info(vars.front(), current_time, dt);

    for (int stage = 0; stage < mesh->blocks.front()->pintg->stages.size(); ++stage) {
      mesh->forward(vars, dt, stage);
    }

    int redo = mesh_redo(mesh, vars);
    if (redo > 0) continue;
    if (redo < 0) break;

    current_time += dt;
    make_outputs(mesh, vars, current_time);
  }

  finalize_outputs(mesh, vars, current_time);
}
