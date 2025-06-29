// snap
#include <snap/snap.h>

#include <snap/mesh/mesh_formatter.hpp>
#include <snap/mesh/meshblock.hpp>
#include <snap/output/output_formats.hpp>

using namespace snap;

int main(int argc, char** argv) {
  auto op = MeshBlockOptions::from_yaml("example_shock.yaml");
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

  w[Index::IDN] = torch::where(x1v < 0, 1.0, 0.125);
  w[Index::IPR] = torch::where(x1v < 0, 1.0, 0.1);

  block->initialize(w);

  // internal boundary
  auto r1 = torch::sqrt(x1v * x1v + x2v * x2v + x3v * x3v);
  auto solid = torch::where(r1 < 0.1, 1, 0);

  // output
  auto out =
      NetcdfOutput(OutputOptions().file_basename("sod").variable("prim"));
  float current_time = 0.;

  out.write_output_file(block, current_time, OctTreeOptions(), 0);
  out.combine_blocks();

  for (int n = 0; n < 200; ++n) {
    auto dt = block->max_time_step(solid);
    for (int stage = 0; stage < block->pintg->stages.size(); ++stage)
      block->forward(dt, stage, solid);

    current_time += dt;
    if ((n + 1) % 10 == 0) {
      std::cout << "time = " << current_time << std::endl;
      ++out.file_number;
      out.write_output_file(block, current_time, OctTreeOptions(), 0);
      out.combine_blocks();
    }
  }
}
