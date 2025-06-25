// snap
#include <snap/snap.h>

#include <snap/mesh/meshblock.hpp>
#include <snap/output/output_formats.hpp>

using namespace snap;

int main(int argc, char** argv) {
  double p0 = 1.E5;
  double Ts = 300.;
  double xc = 0.;
  double xr = 4.e3;
  double zc = 3.e3;
  double zr = 2.e3;
  double dT = -15.;
  double grav = 9.8;
  double Rd = 287.;
  double gamma = 1.4;
  double K = 75.;

  int nx1 = 80;
  int nx2 = 160;
  int nx3 = 160;
  int nghost = 3;

  auto op_coord = CoordinateOptions().nx1(nx1).nx2(nx2).nx3(nx3);
  op_coord.x1min(0).x1max(6.4e3).x2min(0).x2max(25.6e3).x3min(0).x3max(25.6e3);

  auto op_recon =
      ReconstructOptions().interp(InterpOptions("weno5")).shock(false);
  auto op_thermo = kintera::ThermoOptions::from_yaml("xxxx.yaml");
  auto op_eos = EquationOfStateOptions().thermo(op_thermo).type("ideal_gas");
  auto op_riemann = RiemannSolverOptions().type("lmars");
  auto op_grav = ConstGravityOptions().grav1(-grav);
  auto op_proj = PrimitiveProjectorOptions().type("temperature");
  auto op_vic = ImplicitOptions().scheme(1);
  auto op_intg = IntegratorOptions().type("rk3").cfl(0.9);

  auto op_hydro =
      HydroOptions().eos(op_eos).coord(op_coord).riemann(op_riemann);
  op_hydro.recon1(op_recon).recon23(op_recon).grav(op_grav).proj(op_proj).vic(
      op_vic);

  auto op_block = MeshBlockOptions().intg(op_intg).hydro(op_hydro);

  for (int i = 0; i < 6; ++i)
    op_block.bfuncs().push_back(get_bc_func()["reflecting_inner"]);
  auto block = MeshBlock(op_block);

  // block->to(torch::Device(torch::kCUDA, 0));

  // initial conditions
  auto pcoord = block->phydro->pcoord;
  auto peos = block->phydro->peos;

  // thermodynamics
  auto cp = gamma / (gamma - 1.) * Rd;

  /*auto x1v = pcoord->x1v.view({1, 1, -1});
  auto x2v = pcoord->x2v.view({1, -1, 1});
  auto x3v = pcoord->x3v.view({-1, 1, 1});*/

  auto result = torch::meshgrid({pcoord->x3v, pcoord->x2v, pcoord->x1v}, "ij");
  auto x1v = result[2];
  auto x2v = result[1];

  auto const& w = torch::zeros_like(block->phydro->peos->get_buffer("W"));

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

  // block->user_out_var.insert("temp", pthermo->get_temp(w));
  // block->user_out_var.insert("theta", pthermo->get_theta_ref(w, p0));

  out2.write_output_file(block, current_time, OctTreeOptions(), 0);
  out2.combine_blocks();

  // integration
  int n = 0;
  while (true) {
    auto dt = block->max_time_step();
    for (int stage = 0; stage < block->pintg->stages.size(); ++stage) {
      block->forward(dt, stage);
    }

    current_time += dt;
    if ((n + 1) % 100 == 0) {
      std::cout << "time = " << current_time << std::endl;
      ++out2.file_number;
      out2.write_output_file(block, current_time, OctTreeOptions(), 0);
      out2.combine_blocks();
    }

    n++;
    if (current_time > 900) break;
  }
}
