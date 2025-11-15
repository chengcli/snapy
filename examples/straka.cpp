// yaml
#include <yaml-cpp/yaml.h>

// kintera
#include <kintera/constants.h>

#include <kintera/species.hpp>

// snap
#include <snap/snap.h>

#include <snap/eos/ideal_gas.hpp>
#include <snap/mesh/mesh_formatter.hpp>
#include <snap/mesh/meshblock.hpp>
#include <snap/output/output_formats.hpp>

using namespace snap;

int main(int argc, char** argv) {
  auto config = YAML::LoadFile("straka.yaml");

  auto p0 = config["problem"]["p0"].as<double>();
  auto Ts = config["problem"]["Ts"].as<double>();
  auto xc = config["problem"]["xc"].as<double>();
  auto zc = config["problem"]["zc"].as<double>();
  auto xr = config["problem"]["xr"].as<double>();
  auto zr = config["problem"]["zr"].as<double>();
  auto dT = config["problem"]["dT"].as<double>();
  auto K = config["problem"]["K"].as<double>();
  auto grav = -config["forcing"]["const-gravity"]["grav1"].as<double>();

  auto op = MeshBlockOptions::from_yaml("straka.yaml");
  auto block = MeshBlock(op);
  auto device = torch::kCPU;
  if (torch::cuda::is_available()) {
    std::cout << "Running on CUDA" << std::endl;
    device = torch::kCUDA;
  }

  std::cout << fmt::format("MeshBlock Options: {}", block->options)
            << std::endl;

  block->to(device);

  // initial conditions
  auto pcoord = block->phydro->pcoord;
  auto peos = block->phydro->peos;

  // thermodynamics
  auto Rd = kintera::constants::Rgas / kintera::species_weights[0];
  auto cv = kintera::species_cref_R[0] * Rd;
  auto cp = cv + Rd;

  auto grids = torch::meshgrid({pcoord->x3v, pcoord->x2v, pcoord->x1v}, "ij");
  auto x1v = grids[2];
  auto x2v = grids[1];

  int nc1 = pcoord->options.nc1();
  int nc2 = pcoord->options.nc2();
  int nc3 = pcoord->options.nc3();
  int nvar = peos->nvar();

  auto w = torch::zeros(
      {nvar, nc3, nc2, nc1},
      torch::TensorOptions().dtype(torch::kFloat64).device(device));

  auto L = torch::sqrt(((x2v - xc) / xr).square() + ((x1v - zc) / zr).square());

  auto temp = Ts - grav * x1v / cp;

  w[IPR] = p0 * torch::pow(temp / Ts, cp / Rd);
  temp += torch::where(L <= 1, dT * (torch::cos(L * M_PI) + 1.) / 2., 0);
  w[IDN] = w[IPR] / (Rd * temp);

  // initialize
  std::map<std::string, torch::Tensor> vars;
  vars["hydro_w"] = w;
  block->initialize(vars);

  block->user_output_callback = [Rd, cp, p0](Variables const& vars) {
    auto w = vars.at("hydro_w");
    auto temp = w[IPR] / (w[IDN] * Rd);

    Variables out;
    out["temp"] = temp;
    out["theta"] = temp * (p0 / w[IPR]).pow(Rd / cp);
    return out;
  };

  double current_time = 0.;
  block->make_outputs(vars, current_time);

  while (!block->pintg->stop(block->cycle++, current_time)) {
    auto dt = block->max_time_step(vars);
    block->print_cycle_info(vars, current_time, dt);

    for (int stage = 0; stage < block->pintg->stages.size(); ++stage) {
      block->forward(vars, dt, stage);
    }

    current_time += dt;
    block->make_outputs(vars, current_time);
  }
}
