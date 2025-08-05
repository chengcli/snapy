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
  torch::OrderedDict<std::string, torch::Tensor> vars;
  vars.insert("hydro_w", w);
  block->initialize(vars);

  // output
  auto out2 = NetcdfOutput(
      OutputOptions().file_basename("straka").fid(2).variable("prim"));
  auto out3 = NetcdfOutput(
      OutputOptions().file_basename("straka").fid(3).variable("uov"));

  block->user_out_var.insert("temp", temp);
  block->user_out_var.insert("theta", temp * (p0 / w[IPR]).pow(Rd / cp));

  auto m = block->named_modules()["hydro.eos.thermo"];
  auto thermo_y = std::dynamic_pointer_cast<kintera::ThermoYImpl>(m);

  double current_time = 0.;
  int count = 0;
  while (!block->pintg->stop(count, current_time)) {
    auto dt = block->max_time_step(vars);

    if (count % 100 == 0) {
      printf("count = %d, dt = %.6f, time = %.6f\n", count, dt, current_time);
      block->report_timer(std::cout);

      auto ivol =
          thermo_y->compute("DY->V", {w[IDN], w.slice(0, ICY, w.size(0))});
      temp = thermo_y->compute("PV->T", {w[IPR], ivol});

      block->user_out_var["temp"] = temp;
      block->user_out_var["theta"] = temp * (p0 / w[IPR]).pow(Rd / cp);

      out2.write_output_file(block, vars, current_time, OctTreeOptions(), 0);
      out2.combine_blocks();
      out2.file_number++;

      out3.write_output_file(block, vars, current_time, OctTreeOptions(), 0);
      out3.combine_blocks();
      out3.file_number++;
    }

    for (int stage = 0; stage < block->pintg->stages.size(); ++stage) {
      block->forward(dt, stage, vars);
    }

    count++;
    current_time += dt;
  }
}
