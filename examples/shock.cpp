// snap
#include <snap/snap.h>

#include <snap/mesh/mesh_formatter.hpp>
#include <snap/mesh/meshblock.hpp>
#include <snap/output/output_formats.hpp>

using namespace snap;

int main(int argc, char** argv) {
  auto op = MeshBlockOptions::from_yaml("shock.yaml");
  auto block = MeshBlock(op);

  std::cout << fmt::format("MeshBlock Options: {}", block->options)
            << std::endl;

  // block->to(torch::kCUDA);

  // initial conditions
  auto pcoord = block->phydro->pcoord;
  auto peos = block->phydro->peos;

  auto x1v = pcoord->x1v.view({1, 1, -1});
  auto x2v = pcoord->x2v.view({1, -1, 1});
  auto x3v = pcoord->x3v.view({-1, 1, 1});

  auto const& w = block->phydro->peos->get_buffer("W");
  w.zero_();

  w[IDN] = torch::where(x1v < 0, 1.0, 0.125);
  w[IPR] = torch::where(x1v < 0, 1.0, 0.1);

  std::cout << "w shape = " << w.sizes() << std::endl;

  torch::OrderedDict<std::string, torch::Tensor> vars;

  // internal boundary
  auto r1 = torch::sqrt(x1v * x1v + x2v * x2v + x3v * x3v);
  auto solid = torch::where(r1 < 0.1, 1, 0).to(torch::kBool);

  vars.insert("hydro_w", w);
  vars.insert("solid", solid);
  block->initialize(vars);

  // output
  auto out =
      NetcdfOutput(OutputOptions().file_basename("sod").variable("prim"));
  double current_time = 0.;

  int count = 0;
  while (!block->pintg->stop(count, current_time)) {
    auto dt = block->max_time_step(vars);

    if (count % 10 == 0) {
      printf("count = %d, dt = %.6f, time = %.6f\n", count, dt, current_time);
      out.write_output_file(block, vars, current_time, OctTreeOptions(), 0);
      out.combine_blocks();
      out.file_number++;
    }

    for (int stage = 0; stage < block->pintg->stages.size(); ++stage) {
      block->forward(dt, stage, vars);
    }

    count++;
    current_time += dt;
  }
}
