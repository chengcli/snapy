// snap
#include <snap/snap.h>

#include <snap/eos/aneos.hpp>
#include <snap/mesh/mesh_formatter.hpp>
#include <snap/mesh/meshblock.hpp>
#include <snap/output/output_formats.hpp>

using namespace snap;

int main(int argc, char** argv) {
  auto op = MeshBlockOptions::from_yaml("shock_aneos.yaml");
  auto block = MeshBlock(op);

  std::cout << fmt::format("MeshBlock Options: {}", block->options)
            << std::endl;

  block->to(torch::kCUDA);

  // initial conditions
  auto pcoord = block->phydro->pcoord;
  auto peos = block->phydro->peos;

  auto x1v = pcoord->x1v.view({1, 1, -1});
  auto x2v = pcoord->x2v.view({1, -1, 1});
  auto x3v = pcoord->x3v.view({-1, 1, 1});

  int nc1 = pcoord->options.nc1();
  int nc2 = pcoord->options.nc2();
  int nc3 = pcoord->options.nc3();
  int nvar = peos->nvar();

  auto w = torch::zeros(
      {nvar, nc3, nc2, nc1},
      torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCUDA));

  w[IDN] = torch::where(x1v < 0, 1.0 * 1.e3, 0.125 * 1.e3);
  w[IPR] = torch::where(x1v < 0, 40.0 * 1.e9, 1 * 1.e9);

  std::map<std::string, torch::Tensor> vars;
  vars["hydro_w"] = w;
  block->initialize(vars);

  // internal boundary
  auto r1 = torch::sqrt(x1v * x1v + x2v * x2v + x3v * x3v);
  auto solid = torch::where(r1 < 0.1, 1, 0);
  solid.to(torch::kBool);

  // output
  auto out =
      NetcdfOutput(OutputOptions().file_basename("aneos").variable("prim"));
  float current_time = 0.;

  out.write_output_file(block, vars, current_time, 0);
  out.combine_blocks();

  int count = 0;
  while (!block->pintg->stop(count++, current_time)) {
    auto dt = block->max_time_step(vars);
    // auto dt = 1.e-10;
    for (int stage = 0; stage < block->pintg->stages.size(); ++stage)
      block->forward(dt, stage, vars);

    current_time += dt;
    if (count % 100 == 0) {
      block->report_timer(std::cout);
      printf("count = %d, dt = %.6f, time = %.6f\n", count, dt, current_time);
      ++out.file_number;
      out.write_output_file(block, vars, current_time, 0);
      out.combine_blocks();
    }
  }
}
