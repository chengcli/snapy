// C/C++
#include <iomanip>
#include <iostream>
#include <limits>

// snap
#include <snap/input/read_restart_file.hpp>
#include <snap/output/output_formats.hpp>
#include <snap/utils/signal_handler.hpp>

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

MeshBlockImpl::~MeshBlockImpl() {
  // destroy signal handler
  SignalHandler::Destroy();
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
  TORCH_CHECK(phydro->peos->nvar() > 0, "Hydro model must have nvar > 0.");
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
  // Set up a signal handler
  SignalHandler::GetInstance();

  if (pintg->options.restart() != "") {
    read_restart_file(this, pintg->options.restart(), vars);
    return vars;
  }

  BoundaryFuncOptions bops;
  bops.nghost(options.hydro().coord().nghost());

  torch::Tensor hydro_w, scalar_r, solid;

  // hydro
  int64_t nc3 = options.hydro().coord().nc3();
  int64_t nc2 = options.hydro().coord().nc2();
  int64_t nc1 = options.hydro().coord().nc1();

  TORCH_CHECK(vars.count("hydro_w"),
              "initialize: hydro_w is required for hydro model.");
  hydro_w = vars.at("hydro_w");
  TORCH_CHECK(hydro_w.sizes() ==
                  std::vector<int64_t>({phydro->peos->nvar(), nc3, nc2, nc1}),
              "initialize: hydro_w has incorrect shape.", " Expected [",
              phydro->peos->nvar(), ", ", nc3, ", ", nc2, ", ", nc1,
              "] but got ", hydro_w.sizes());
  vars["hydro_u"] = phydro->peos->compute("W->U", {hydro_w});

  bops.type(kConserved);
  for (int i = 0; i < options.bfuncs().size(); ++i) {
    if (options.bfuncs()[i] == nullptr) continue;
    options.bfuncs()[i](vars.at("hydro_u"), 3 - i / 2, bops);
  }

  phydro->peos->forward(vars.at("hydro_u"), /*out=*/hydro_w);

  // scalar
  if (pscalar->nvar() > 0) {
    TORCH_CHECK(vars.count("scalar_r"),
                "initialize: scalar_r is required for scalar model.");
    scalar_r = vars.at("scalar_r");
    TORCH_CHECK(scalar_r.sizes() ==
                    std::vector<int64_t>({pscalar->nvar(), nc3, nc2, nc1}),
                "initialize: scalar_r has incorrect shape.", " Expected [",
                pscalar->nvar(), ", ", nc3, ", ", nc2, ", ", nc1, "] but got ",
                scalar_r.sizes());
    vars["scalar_s"] = hydro_w[IDN] * scalar_r;

    bops.type(kScalar);
    for (int i = 0; i < options.bfuncs().size(); ++i) {
      if (options.bfuncs()[i] == nullptr) continue;
      options.bfuncs()[i](vars.at("scalar_s"), 3 - i / 2, bops);
    }
  }

  // solid
  if (vars.count("solid")) {
    solid = vars.at("solid");
    TORCH_CHECK(solid.sizes() == std::vector<int64_t>({nc3, nc2, nc1}),
                "initialize: solid has incorrect shape.", " Expected [", nc3,
                ", ", nc2, ", ", nc1, "] but got ", solid.sizes());
    vars["fill_solid_hydro_w"] =
        torch::where(solid.unsqueeze(0).expand_as(hydro_w), hydro_w, 0.);
    vars["fill_solid_hydro_w"].narrow(0, IVX, 3).zero_();
    phydro->pib->mark_prim_solid_(hydro_w, solid);

    vars["fill_solid_hydro_u"] =
        torch::where(solid.unsqueeze(0).expand_as(vars.at("hydro_u")),
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

  // hyperbolic hydro time step
  if (vars.count("solid")) {
    dt = std::min(dt, phydro->max_time_step(w, vars.at("solid")));
  } else {
    dt = std::min(dt, phydro->max_time_step(w));
  }

  return pow(2., -pintg->current_redo) * pintg->options.cfl() * dt;
}

void MeshBlockImpl::forward(Variables& vars, double dt, int stage) {
  TORCH_CHECK(stage >= 0 && stage < pintg->stages.size(),
              "Invalid stage: ", stage);

  auto hydro_u = vars.at("hydro_u");
  auto scalar_s = vars.count("scalars") ? vars.at("scalar_s") : torch::Tensor();

  auto start = std::chrono::high_resolution_clock::now();
  // -------- (1) save initial state --------
  if (stage == 0) {
    _hydro_u0.copy_(hydro_u);
    _hydro_u1.copy_(hydro_u);

    if (pscalar->nvar() > 0) {
      _scalar_s0.copy_(scalar_s);
      _scalar_s1.copy_(scalar_s);
    }
  }

  // -------- (2) set containers for future results --------
  torch::Tensor fut_hydro_du, fut_scalar_ds;

  // -------- (3) launch all jobs --------
  // (3.1) hydro forward
  fut_hydro_du = phydro->forward(dt, hydro_u, vars);

  // (3.2) scalar forward
  if (pscalar->nvar() > 0) {
    fut_scalar_ds = pscalar->forward(dt, scalar_s, vars);
  }

  // -------- (4) multi-stage averaging --------
  hydro_u.set_(pintg->forward(stage, _hydro_u0, _hydro_u1, fut_hydro_du));
  _hydro_u1.copy_(hydro_u);

  if (pscalar->nvar() > 0) {
    scalar_s.set_(pintg->forward(stage, _scalar_s0, _scalar_s1, fut_scalar_ds));
    _scalar_s1.copy_(scalar_s);
  }

  // -------- (5) update ghost zones --------
  BoundaryFuncOptions bops;
  bops.nghost(options.hydro().coord().nghost());

  if (vars.count("solid")) {
    phydro->pib->fill_cons_solid_(hydro_u, vars.at("solid"),
                                  vars.at("fill_solid_hydro_u"));
  }

  // (5.1) apply hydro boundary
  bops.type(kConserved);
  for (int i = 0; i < options.bfuncs().size(); ++i) {
    if (options.bfuncs()[i] == nullptr) continue;
    options.bfuncs()[i](hydro_u, 3 - i / 2, bops);
  }

  // (5.2) apply scalar boundary
  if (pscalar->nvar() > 0) {
    bops.type(kScalar);
    for (int i = 0; i < options.bfuncs().size(); ++i) {
      if (options.bfuncs()[i] == nullptr) continue;
      options.bfuncs()[i](scalar_s, 3 - i / 2, bops);
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
}

void MeshBlockImpl::make_outputs(Variables const& vars, double current_time,
                                 bool final_write) {
  for (auto& output_type : output_types) {
    if (final_write) {
      output_type->write_output_file(this, vars, current_time, final_write);
    } else if (current_time >= output_type->next_time) {
      output_type->write_output_file(this, vars, current_time, final_write);
      output_type->next_time += output_type->options.dt();
      output_type->file_number += 1;
    }
  }
}

void MeshBlockImpl::print_cycle_info(Variables const& vars, double time,
                                     double dt) const {
  if (options.dist().gid() != 0) return;  // only rank 0 prints

  const int dt_precision = std::numeric_limits<double>::max_digits10 - 3;
  bool compute_mass = false;
  bool compute_energy = false;

  if (pintg->options.ncycle_out() != 0) {
    if (cycle % pintg->options.ncycle_out() == 0) {
      if (vars.count("hydro_u")) {
        compute_mass = true;
        compute_energy = true;
      }
      std::cout << "cycle=" << cycle << " redo=" << pintg->current_redo
                << std::scientific << std::setprecision(dt_precision)
                << " time=" << time << " dt=" << dt;
      auto interior = part({0, 0, 0});
      if (compute_mass) {
        auto mass =
            vars.at("hydro_u").index(interior)[IDN].sum().item<double>();
        std::cout << " mass=" << mass;
      }
      if (compute_energy) {
        auto energy =
            vars.at("hydro_u").index(interior)[IPR].sum().item<double>();
        std::cout << " energy=" << energy;
      }
      std::cout << std::endl;
    }
  }
}

void MeshBlockImpl::finalize(Variables const& vars, double time) {
  // make final output
  make_outputs(vars, time, /*final_write=*/true);

  if (options.dist().gid() == 0) {  // only rank 0 prints
    auto sig = SignalHandler::GetInstance();
    if (sig->GetSignalFlag(SIGTERM) != 0) {
      std::cout << std::endl << "Terminating on Terminate signal" << std::endl;
    } else if (sig->GetSignalFlag(SIGINT) != 0) {
      std::cout << std::endl << "Terminating on Interrupt signal" << std::endl;
    } else if (sig->GetSignalFlag(SIGALRM) != 0) {
      std::cout << std::endl << "Terminating on wall-time limit" << std::endl;
    } else if (pintg->options.nlim() >= 0 && cycle >= pintg->options.nlim()) {
      std::cout << std::endl << "Terminating on cycle limit" << std::endl;
    } else if (time >= pintg->options.tlim()) {
      std::cout << std::endl << "Terminating on time limit" << std::endl;
    } else {
      std::cout << std::endl << "Terminating abnormally" << std::endl;
    }

    std::cout << "time=" << time << " cycle=" << cycle - 1 << std::endl;
    std::cout << "tlim=" << pintg->options.tlim()
              << " nlim=" << pintg->options.nlim() << std::endl;
  }
}

int MeshBlockImpl::check_redo(Variables& vars) {
  auto sig = snap::SignalHandler::GetInstance();
  if (sig->CheckSignalFlags()) return -1;  // terminate

  // check if density or pressure is negative
  auto hydro_u = vars.at("hydro_u");
  auto interior = part({0, 0, 0});
  auto rho = hydro_u.index(interior)[IDN];
  auto pres = hydro_u.index(interior)[IPR];

  if (rho.min().item<double>() <= 0. || pres.min().item<double>() <= 0.) {
    std::cout << "Negative density/pressure detected. Redoing the step with "
                 "smaller dt."
              << std::endl;
    pintg->current_redo += 1;
    if (pintg->current_redo > pintg->options.max_redo()) {
      std::cout << "Maximum number of redo attempts exceeded. Terminating."
                << std::endl;
      return -1;  // terminate
    }

    // reset variables
    vars["hydro_u"].copy_(_hydro_u0);
    if (vars.count("scalar_s")) {
      vars["scalar_s"].copy_(_scalar_s0);
    }

    // reset cycle
    cycle -= 1;
    return 1;  // redo
  }

  // good to go
  pintg->current_redo = 0;
  return 0;
}

}  // namespace snap
