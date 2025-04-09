// snap
#include <snap/snap.h>

#include <snap/mesh/meshblock.hpp>
#include <snap/output/output_formats.hpp>

using namespace snap;

int main(int argc, char **argv) {
  int nx1 = 256;
  int nx2 = 256;
  int nx3 = 1;
  int nghost = 3;

  auto op_coord = CoordinateOptions();
  op_coord.nx1(nx1).nx2(nx2);
  op_coord.x1min(-0.5).x1max(0.5);
  op_coord.x2min(-0.5).x2max(0.5);
  op_coord.x3min(-0.5).x3max(0.5);

  auto op_recon =
      ReconstructOptions().interp(InterpOptions("weno5")).shock(true);
  auto op_eos = EquationOfStateOptions();
  auto op_riemann = RiemannSolverOptions().type("hllc");
  auto op_intg = IntegratorOptions().type("rk3").cfl(0.9);

  auto op_hydro =
      HydroOptions().eos(op_eos).coord(op_coord).riemann(op_riemann);
  op_hydro.recon1(op_recon).recon23(op_recon);

  auto op_block =
      MeshBlockOptions().nghost(nghost).intg(op_intg).hydro(op_hydro);

  for (int i = 0; i < 4; ++i)
    op_block.bflags().push_back(BoundaryFlag::kOutflow);

  auto block = MeshBlock(op_block);

  // block->to(torch::kCUDA);

  // initial conditions
  auto pcoord = block->phydro->pcoord;
  auto peos = block->phydro->peos;

  auto x1v = pcoord->x1v.view({1, 1, -1});
  auto x2v = pcoord->x2v.view({1, -1, 1});
  auto x3v = pcoord->x3v.view({-1, 1, 1});

  auto w = torch::zeros_like(block->hydro_u);
  w[Index::IDN] = torch::where(x1v < 0, 1.0, 0.125);
  w[Index::IPR] = torch::where(x1v < 0, 1.0, 0.1);
  w[Index::IVX] = w[Index::IVY] = w[Index::IVZ] = 0.0;

  block->set_primitives(w);

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
    auto dt = block->max_root_time_step(0, solid);
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
