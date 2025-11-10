// C/C++
#include <iomanip>
#include <iostream>
#include <limits>

// snap
#include <snap/input/read_restart_file.hpp>
#include <snap/output/output_formats.hpp>

#include "meshblock.hpp"

namespace snap {

MeshBlockImpl::MeshBlockImpl(MeshBlockOptions const& options_)
    : options(std::move(options_)) {
  int nc1 = options.hydro().coord().nc1();
  int nc2 = options.hydro().coord().nc2();
  int nc3 = options.hydro().coord().nc3();

  if (nc1 > 1 && options.bfuncs().size() < 2) {
    throw std::runtime_error("MeshBlockImpl: bfuncs size must be at least 2");
  }

  if (nc2 > 1 && options.bfuncs().size() < 4) {
    throw std::runtime_error("MeshBlockImpl: bfuncs size must be at least 4");
  }

  if (nc3 > 1 && options.bfuncs().size() < 6) {
    throw std::runtime_error("MeshBlockImpl: bfuncs size must be at least 6");
  }

  reset();
}

void MeshBlockImpl::reset() {
  // set up output
  for (auto const& out_op : options.outputs()) {
    if (out_op.file_type() == "restart") {
      output_types.push_back(std::make_shared<RestartOutput>(out_op));
    } else if (out_op.file_type() == "netcdf") {
      output_types.push_back(std::make_shared<NetcdfOutput>(out_op));
      /*} else if (out_op.file_type() == "hdf5") {
        output_types.push_back(
            std::make_shared<HDF5Output>(out_op));*/
    } else {
      throw std::runtime_error("Output type '" + out_op.file_type() +
                               "' is not implemented.");
    }
  }

  // set up integrator
  pintg = register_module("intg", Integrator(options.intg()));
  options.intg() = pintg->options;

  // set up hydro model
  phydro = register_module("hydro", Hydro(options.hydro()));
  options.hydro() = phydro->options;

  // set up scalar model
  pscalar = register_module("scalar", Scalar(options.scalar()));
  options.scalar() = pscalar->options;

  // dimensions
  int nc1 = options.hydro().coord().nc1();
  int nc2 = options.hydro().coord().nc2();
  int nc3 = options.hydro().coord().nc3();
  auto peos = phydro->peos;

  // set up hydro buffer
  _hydro_u0 = register_buffer(
      "u0",
      torch::zeros({phydro->peos->nvar(), nc3, nc2, nc1}, torch::kFloat64));
  _hydro_u1 = register_buffer(
      "u1",
      torch::zeros({phydro->peos->nvar(), nc3, nc2, nc1}, torch::kFloat64));

  // set up scalar buffer
  _scalar_s0 = register_buffer(
      "s0", torch::zeros({pscalar->nvar(), nc3, nc2, nc1}, torch::kFloat64));
  _scalar_s1 = register_buffer(
      "s1", torch::zeros({pscalar->nvar(), nc3, nc2, nc1}, torch::kFloat64));
}

std::vector<torch::indexing::TensorIndex> MeshBlockImpl::part(
    std::tuple<int, int, int> offset, bool exterior, int extend_x1,
    int extend_x2, int extend_x3) const {
  int nc1 = options.hydro().coord().nc1();
  int nc2 = options.hydro().coord().nc2();
  int nc3 = options.hydro().coord().nc3();
  int nghost_coord = options.hydro().coord().nghost();

  int is_ghost = exterior ? 1 : 0;

  auto [o3, o2, o1] = offset;
  int start1, len1, start2, len2, start3, len3;

  int nx1 = nc1 > 1 ? nc1 - 2 * nghost_coord : 1;
  int nx2 = nc2 > 1 ? nc2 - 2 * nghost_coord : 1;
  int nx3 = nc3 > 1 ? nc3 - 2 * nghost_coord : 1;

  // ---- dimension 1 ---- //
  int nghost = nx1 == 1 ? 0 : nghost_coord;

  if (o1 == -1) {
    start1 = nghost * (1 - is_ghost);
    len1 = nghost;
  } else if (o1 == 0) {
    start1 = nghost;
    len1 = nx1 + extend_x1;
  } else {  // o1 == 1
    start1 = nghost * is_ghost + nx1;
    len1 = nghost;
  }

  // ---- dimension 2 ---- //
  nghost = nx2 == 1 ? 0 : nghost_coord;

  if (o2 == -1) {
    start2 = nghost * (1 - is_ghost);
    len2 = nghost;
  } else if (o2 == 0) {
    start2 = nghost;
    len2 = nx2 + extend_x2;
  } else {  // o2 == 1
    start2 = nghost * is_ghost + nx2;
    len2 = nghost;
  }

  // ---- dimension 3 ---- //
  nghost = nx3 == 1 ? 0 : nghost_coord;

  if (o3 == -1) {
    start3 = nghost * (1 - is_ghost);
    len3 = nghost;
  } else if (o3 == 0) {
    start3 = nghost;
    len3 = nx3 + extend_x3;
  } else {  // o3 == 1
    start3 = nghost * is_ghost + nx3;
    len3 = nghost;
  }

  auto slice1 = torch::indexing::Slice(start1, start1 + len1);
  auto slice2 = torch::indexing::Slice(start2, start2 + len2);
  auto slice3 = torch::indexing::Slice(start3, start3 + len3);
  auto slice4 = torch::indexing::Slice();

  return {slice4, slice3, slice2, slice1};
}

Variables& MeshBlockImpl::initialize(Variables& vars) {
  if (pintg->options.restart() != "") {
    read_restart_file(this, pintg->options.restart(), vars);
    return vars;
  }

  BoundaryFuncOptions op;
  op.nghost(options.hydro().coord().nghost());

  if (!vars.count("hydro_u")) {
    vars["hydro_u"] = torch::Tensor();
  }

  if (!vars.count("scalar_s")) {
    vars["scalar_s"] = torch::Tensor();
  }

  if (!vars.count("scalar_r")) {
    vars["scalar_r"] = torch::Tensor();
  }

  if (!vars.count("solid")) {
    vars["solid"] = torch::Tensor();
  }

  auto& hydro_w = vars.at("hydro_w");
  auto& scalar_r = vars.at("scalar_r");

  // hydro
  if (phydro->peos->nvar() > 0) {
    vars["hydro_u"] = phydro->peos->compute("W->U", {hydro_w});

    op.type(kConserved);
    for (int i = 0; i < options.bfuncs().size(); ++i) {
      if (options.bfuncs()[i] == nullptr) continue;
      options.bfuncs()[i](vars.at("hydro_u"), 3 - i / 2, op);
    }

    phydro->peos->forward(vars.at("hydro_u"), /*out=*/hydro_w);
  }

  // scalar
  if (pscalar->nvar() > 0) {
    auto temp = phydro->peos->compute("W->T", {hydro_w});
    vars.at("scalar_s") = pscalar->pthermo->compute(
        "TPX->V", {temp, hydro_w[Index::IPR], scalar_r});

    op.type(kScalar);
    for (int i = 0; i < options.bfuncs().size(); ++i) {
      if (options.bfuncs()[i] == nullptr) continue;
      options.bfuncs()[i](vars.at("scalar_s"), 3 - i / 2, op);
    }

    // FIXME: scalar should have an eos as well
    // scalar_r.set_(pscalar->pthermo->compute("V->X", {scalar_s}));
  }

  // solid
  if (vars["solid"].defined()) {
    vars["fill_solid_hydro_w"] = torch::where(
        vars.at("solid").unsqueeze(0).expand_as(hydro_w), hydro_w, 0.);
    vars["fill_solid_hydro_w"].narrow(0, IVX, 3).zero_();
    phydro->pib->mark_prim_solid_(hydro_w, vars.at("solid"));

    vars["fill_solid_hydro_u"] = torch::where(
        vars.at("solid").unsqueeze(0).expand_as(vars.at("hydro_u")),
        vars.at("hydro_u"), 0.);
    vars["fill_solid_hydro_u"].narrow(0, IVX, 3).zero_();
  } else {
    vars["fill_solid_hydro_w"] = hydro_w;
    vars["fill_solid_hydro_u"] = vars.at("hydro_u");
  }

  return vars;
}

double MeshBlockImpl::max_time_step(Variables const& vars) {
  double dt = 1.e9;

  auto const& w = vars.at("hydro_w");

  if (phydro->peos->nvar() > 0) {
    dt = std::min(dt, phydro->max_time_step(w, vars.at("solid")));
  }

  return pintg->options.cfl() * dt;
}

Variables& MeshBlockImpl::forward(double dt, int stage, Variables& vars) {
  TORCH_CHECK(stage >= 0 && stage < pintg->stages.size(),
              "Invalid stage: ", stage);

  auto& hydro_u = vars.at("hydro_u");
  auto& scalar_s = vars.at("scalar_s");

  auto start = std::chrono::high_resolution_clock::now();
  // -------- (1) save initial state --------
  if (stage == 0) {
    if (phydro->peos->nvar() > 0) {
      _hydro_u0.copy_(hydro_u);
      _hydro_u1.copy_(hydro_u);
    }

    if (pscalar->nvar() > 0) {
      _scalar_s0.copy_(scalar_s);
      _scalar_s1.copy_(scalar_s);
    }
  }

  // -------- (2) set containers for future results --------
  torch::Tensor fut_hydro_du, fut_scalar_ds;

  // -------- (3) launch all jobs --------
  // (3.1) hydro forward
  if (phydro->peos->nvar() > 0) {
    fut_hydro_du = phydro->forward(dt, hydro_u, vars);
  }

  // (3.2) scalar forward
  if (pscalar->nvar() > 0) {
    fut_scalar_ds = pscalar->forward(dt, scalar_s, vars);
  }

  // -------- (4) multi-stage averaging --------
  if (phydro->peos->nvar() > 0) {
    hydro_u.set_(pintg->forward(stage, _hydro_u0, _hydro_u1, fut_hydro_du));
    _hydro_u1.copy_(hydro_u);
  }

  if (pscalar->nvar() > 0) {
    scalar_s.set_(pintg->forward(stage, _scalar_s0, _scalar_s1, fut_scalar_ds));
    _scalar_s1.copy_(scalar_s);
  }

  // -------- (5) update ghost zones --------
  BoundaryFuncOptions op;
  op.nghost(options.hydro().coord().nghost());

  phydro->pib->fill_cons_solid_(hydro_u, vars.at("solid"),
                                vars.at("fill_solid_hydro_u"));

  // (5.1) apply hydro boundary
  if (phydro->peos->nvar() > 0) {
    op.type(kConserved);
    for (int i = 0; i < options.bfuncs().size(); ++i) {
      if (options.bfuncs()[i] == nullptr) continue;
      options.bfuncs()[i](hydro_u, 3 - i / 2, op);
    }
  }

  // (5.2) apply scalar boundary
  if (pscalar->nvar() > 0) {
    op.type(kScalar);
    for (int i = 0; i < options.bfuncs().size(); ++i) {
      if (options.bfuncs()[i] == nullptr) continue;
      options.bfuncs()[i](scalar_s, 3 - i / 2, op);
    }
  }

  // -------- (6) saturation adjustment --------
  if (stage == pintg->stages.size() - 1 &&
      (phydro->options.eos().type() == "ideal-moist" ||
       phydro->options.eos().type() == "moist-mixture")) {
    phydro->peos->apply_conserved_limiter_(hydro_u);

    int ny = hydro_u.size(0) - 5;  // number of species

    auto ke = phydro->peos->compute("U->K", {hydro_u});
    auto rho = hydro_u[IDN] + hydro_u.narrow(0, ICY, ny).sum(0);
    auto ie = hydro_u[Index::IPR] - ke;

    auto yfrac = hydro_u.narrow(0, Index::ICY, ny) / rho;

    auto m = named_modules()["hydro.eos.thermo"];
    auto pthermo = std::dynamic_pointer_cast<kintera::ThermoYImpl>(m);

    pthermo->forward(rho, ie, yfrac, /*warm_start=*/true);

    hydro_u.narrow(0, Index::ICY, ny) = yfrac * rho;
  }

  return vars;
}

void MeshBlockImpl::make_outputs(Variables const& vars, double current_time,
                                 bool force_write) {
  for (auto& output_type : output_types) {
    if (current_time >= output_type->next_time) {
      output_type->write_output_file(this, vars, current_time, force_write);

      // Update next_time and file_number
      output_type->next_time += output_type->options.dt();
      output_type->file_number += 1;
    }
  }
}

void MeshBlockImpl::print_cycle_info(double time, double dt) const {
  if (options.dist().gid() != 0) return;  // only rank 0 prints

  const int dt_precision = std::numeric_limits<double>::max_digits10 - 1;
  if (pintg->options.ncycle_out() != 0) {
    if (cycle % pintg->options.ncycle_out() == 0) {
      std::cout << "cycle=" << cycle << std::scientific
                << std::setprecision(dt_precision) << " time=" << time
                << " dt=" << dt;
      std::cout << std::endl;
    }
  }
}

}  // namespace snap
