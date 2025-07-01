// yaml
#include <yaml-cpp/yaml.h>

// snap
#include <snap/snap.h>

#include <snap/mesh/meshblock.hpp>
#include <snap/output/output_formats.hpp>

using namespace snap;

int main(int argc, char** argv) {
  auto config = YAML::LoadFile("example_straka.yaml");

  auto p0 = config["problem"]["p0"].as<double>();
  auto Ts = config["problem"]["Ts"].as<double>();
  auto xc = config["problem"]["xc"].as<double>();
  auto zc = config["problem"]["zc"].as<double>();
  auto xr = config["problem"]["xr"].as<double>();
  auto zr = config["problem"]["zr"].as<double>();
  auto dT = config["problem"]["dT"].as<double>();
  auto K = config["problem"]["K"].as<double>();

  auto op = MeshBlockOptions::from_yaml("example_straka.yaml");
  auto block = MeshBlock(op);

  std::cout << fmt::format("MeshBlock Options: {}", block->options)
            << std::endl;

  block->to(torch::kCUDA);

  // initial conditions
  auto pcoord = block->phydro->pcoord;
  auto peos = block->phydro->peos;

  // thermodynamics
  auto cp = gamma / (gamma - 1.) * Rd;

  /*auto x1v = pcoord->x1v.view({1, 1, -1});
  auto x2v = pcoord->x2v.view({1, -1, 1});
  auto x3v = pcoord->x3v.view({-1, 1, 1});*/

  auto [x3v, x2v, x1v] =
      torch::meshgrid({pcoord->x3v, pcoord->x2v, pcoord->x1v}, "ij");
  // auto x1v = result[2];
  // auto x2v = result[1];

  auto const& w = block->phydro->peos->get_buffer("W");
  w.zero_();

  auto L = torch::sqrt(((x1v - xc) / xr).square() + ((x2v - zc) / zr).square());

  auto temp = Ts - grav * x1v / cp;

  w[Index::IPR] = p0 * torch::pow(temp / Ts, cp / Rd);
  temp += torch::where(L <= 1, dT * (torch::cos(L * M_PI) + 1.) / 2., 0);
  w[Index::IDN] = w[Index::IPR] / (Rd * temp);

  block->initialize(w);

  // output
  auto out2 = NetcdfOutput(
      OutputOptions().file_basename("straka").fid(2).variable("prim"));
  auto out3 = NetcdfOutput(
      OutputOptions().file_basename("straka").fid(3).variable("uov"));
  double current_time = 0.;

  out2.write_output_file(block, current_time, OctTreeOptions(), 0);
  out2.combine_blocks();

  int count = 0;
  while (!block->pintg->stop(count++, current_time)) {
    auto dt = block->max_time_step();
    for (int stage = 0; stage < block->pintg->stages.size(); ++stage) {
      block->forward(dt, stage);
    }

    current_time += dt;
    if ((n + 1) % 100 == 0) {
      printf("count = %d, dt = %.6f, time = %.6f\n", count, dt, current_time);
      ++out2.file_number;
      out2.write_output_file(block, current_time, OctTreeOptions(), 0);
      out2.combine_blocks();
    }
  }
}
