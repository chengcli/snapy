// C/C++
#include <ctime>
#include <iomanip>
#include <iostream>
#include <limits>
#include <mutex>

// kintera
#include <kintera/utils/serialize.hpp>

// snap
#include <snap/output/output_formats.hpp>
#include <snap/utils/log.hpp>
#include <snap/utils/signal_handler.hpp>

#include "meshblock.hpp"

namespace snap {

static std::mutex meshblock_mutex;

// Static member variable definitions
Layout MeshBlockImpl::_playout = nullptr;

MeshBlockImpl::MeshBlockImpl(MeshBlockOptions const& options_)
    : options(options_) {
  int nc1 = options->coord()->nc1();
  int nc2 = options->coord()->nc2();
  int nc3 = options->coord()->nc3();

  if (nc1 > 1 && options->bfuncs().size() < 2) {
    throw std::runtime_error("MeshBlockImpl: bfuncs size must be at least 2");
  }

  if (nc2 > 1 && options->bfuncs().size() < 4) {
    throw std::runtime_error("MeshBlockImpl: bfuncs size must be at least 4");
  }

  if (nc3 > 1 && options->bfuncs().size() < 6) {
    throw std::runtime_error("MeshBlockImpl: bfuncs size must be at least 6");
  }

  reset();
}

MeshBlockImpl::~MeshBlockImpl() {
  // destroy signal handler
  SignalHandler::Destroy();
}

void MeshBlockImpl::reset() {
  //// ---- (1) set up distributed environment ---- ////
  if (_playout == nullptr) {
    std::unique_lock<std::mutex> lock(meshblock_mutex);
    if (_playout == nullptr) {  // Check again after acquiring lock
      _playout = LayoutImpl::create(options->layout(), this);
    }
  }

  int px = options->layout()->px();
  int py = options->layout()->py();
  int pz = options->layout()->pz();

  int nranks = px * py * pz;
  if (options->layout()->type() == "cubed-sphere") {
    nranks *= 6;
  }
  int rank = options->layout()->rank();

  TORCH_CHECK(options->layout()->world_size() == nranks,
              "MeshBlockImpl: world_size (", options->layout()->world_size(),
              ") does not match layout partitioning (", nranks, ").");

  //// ---- (2) reset internal block boundaries ---- ////
  if (options->layout()->type() != "cubed-sphere") {  // slab or cubed layout
    auto [lx2, lx3, lx1] = _playout->loc_of(rank);

    // x1-dir
    if ((lx1 != 0) || options->layout()->periodic_z()) {
      options->bfuncs()[BoundaryFace::kInnerX1] = nullptr;
    }

    if ((lx1 != pz - 1) || options->layout()->periodic_z()) {
      options->bfuncs()[BoundaryFace::kOuterX1] = nullptr;
    }

    // x2-dir
    if ((lx2 != 0) || options->layout()->periodic_x()) {
      options->bfuncs()[BoundaryFace::kInnerX2] = nullptr;
    }

    if ((lx2 != px - 1) || options->layout()->periodic_x()) {
      options->bfuncs()[BoundaryFace::kOuterX2] = nullptr;
    }

    // x3-dir
    if ((lx3 != 0) || options->layout()->periodic_y()) {
      options->bfuncs()[BoundaryFace::kInnerX3] = nullptr;
    }

    if ((lx3 != py - 1) || options->layout()->periodic_y()) {
      options->bfuncs()[BoundaryFace::kOuterX3] = nullptr;
    }

    if (options->verbose()) {
      SINFO(MeshBlock) << "setting up rank bcs" << std::endl;
      for (int i = 0; i < options->bfuncs().size(); ++i) {
        if (options->bfuncs()[i] == nullptr) {
          SINFO(MeshBlock) << "  bc func " << i << ": internal/custom"
                           << std::endl;
        } else {
          SINFO(MeshBlock) << "  bc func " << i << ": external" << std::endl;
        }
      }
    }
  }

  //// --------- (3) set up output --------- ////
  for (auto const& out_op : options->outputs()) {
    if (out_op->file_type() == "restart") {
      output_types.push_back(std::make_shared<RestartOutput>(out_op));
    } else if (out_op->file_type() == "netcdf") {
      output_types.push_back(std::make_shared<NetcdfOutput>(out_op));
      /*} else if (out_op.file_type() == "hdf5") {
        output_types.push_back(
            std::make_shared<HDF5Output>(out_op));*/
    } else {
      throw std::runtime_error("Output type '" + out_op->file_type() +
                               "' is not implemented.");
    }

    if (options->verbose()) {
      SINFO(MeshBlock) << "adding output type: " << out_op->file_type()
                       << std::endl;
    }
  }

  //// -------- (4) set up integrator -------- ////
  pintg = harp::IntegratorImpl::create(options->intg(), this);
  if (options->verbose()) {
    SINFO(MeshBlock) << "using integrator type: " << pintg->options->type()
                     << std::endl;
  }

  //// ----- (5) set up coordinate model ------ ////
  pcoord = CoordinateImpl::create(options->coord(), this);
  if (options->verbose()) {
    SINFO(MeshBlock) << "using coordinate type: " << pcoord->options->type()
                     << "\n";
  }

  //// -------- (6) set up hydro model -------- ////
  phydro = HydroImpl::create(options->hydro(), this);
  if (options->verbose()) {
    SINFO(MeshBlock) << "using hydro type: " << phydro->peos->options->type()
                     << std::endl;
  }

  //// -------- (7) set up scalar model ------- ////
  pscalar = ScalarImpl::create(options->scalar(), this);

  //// ------ (8) set up internal boundary ---- ////
  pib = InternalBoundaryImpl::create(options->ib(), this);
  if (options->verbose()) {
    SINFO(MeshBlock) << "Internal boundary max-iter: "
                     << pib->options->max_iter() << "\n";
  }

  // dimensions
  int nc1 = options->coord()->nc1();
  int nc2 = options->coord()->nc2();
  int nc3 = options->coord()->nc3();
  auto peos = phydro->peos;

  //// ---------- (9) set up hydro buffer ------ ////
  TORCH_CHECK(phydro->peos->nvar() > 0, "Hydro model must have nvar > 0.");
  _hydro_u0 = register_buffer(
      "u0",
      torch::zeros({phydro->peos->nvar(), nc3, nc2, nc1}, torch::kFloat64));
  _hydro_u1 = register_buffer(
      "u1",
      torch::zeros({phydro->peos->nvar(), nc3, nc2, nc1}, torch::kFloat64));

  //// ------- (10) set up scalar buffer ------- ////
  _scalar_s0 = register_buffer(
      "s0", torch::zeros({pscalar->nvar(), nc3, nc2, nc1}, torch::kFloat64));
  _scalar_s1 = register_buffer(
      "s1", torch::zeros({pscalar->nvar(), nc3, nc2, nc1}, torch::kFloat64));

  if (options->verbose()) {
    SINFO(MeshBlock) << "setting up buffer with shapes:" << std::endl
                     << "* hydro_u0: " << _hydro_u0.sizes() << std::endl
                     << "* hydro_u1: " << _hydro_u1.sizes() << std::endl
                     << "* scalar_s0: " << _scalar_s0.sizes() << std::endl
                     << "* scalar_s1: " << _scalar_s1.sizes() << std::endl;
  }
}

std::vector<torch::indexing::TensorIndex> MeshBlockImpl::part(
    std::tuple<int, int, int> offset, PartOptions const& opts) const {
  int nc1 = options->coord()->nc1();
  int nc2 = options->coord()->nc2();
  int nc3 = options->coord()->nc3();
  int nghost_coord = options->coord()->nghost();

  int is_ghost = opts.exterior() ? 1 : 0;

  auto [o3, o2, o1] = offset;
  int start1, len1, start2, len2, start3, len3;

  int nx1 = nc1 > 1 ? nc1 - 2 * nghost_coord : 1;
  int nx2 = nc2 > 1 ? nc2 - 2 * nghost_coord : 1;
  int nx3 = nc3 > 1 ? nc3 - 2 * nghost_coord : 1;

  // ---- dimension 1 ---- //
  int nghost = nx1 == 1 ? 0 : nghost_coord;

  if (o1 == -1) {
    start1 = nghost * (1 - is_ghost);
    len1 = std::min(nghost, opts.depth());
  } else if (o1 == 0) {
    start1 = nghost - opts.extend_x1();
    len1 = nx1 + 2 * opts.extend_x1();
  } else {  // o1 == 1
    start1 = nx1 + nghost * is_ghost;
    len1 = std::min(nghost, opts.depth());
  }

  // ---- dimension 2 ---- //
  nghost = nx2 == 1 ? 0 : nghost_coord;

  if (o2 == -1) {
    start2 = nghost * (1 - is_ghost);
    len2 = std::min(nghost, opts.depth());
  } else if (o2 == 0) {
    start2 = nghost - opts.extend_x2();
    len2 = nx2 + 2 * opts.extend_x2();
  } else {  // o2 == 1
    start2 = nx2 + nghost * is_ghost;
    len2 = std::min(nghost, opts.depth());
  }

  // ---- dimension 3 ---- //
  nghost = nx3 == 1 ? 0 : nghost_coord;

  if (o3 == -1) {
    start3 = nghost * (1 - is_ghost);
    len3 = std::min(nghost, opts.depth());
  } else if (o3 == 0) {
    start3 = nghost - opts.extend_x3();
    len3 = nx3 + 2 * opts.extend_x3();
  } else {  // o3 == 1
    start3 = nx3 + nghost * is_ghost;
    len3 = std::min(nghost, opts.depth());
  }

  auto slice1 = torch::indexing::Slice(start1, start1 + len1);
  auto slice2 = torch::indexing::Slice(start2, start2 + len2);
  auto slice3 = torch::indexing::Slice(start3, start3 + len3);
  auto slice4 = torch::indexing::Slice();

  if (opts.ndim() == 3) {
    return {slice3, slice2, slice1};
  } else if (opts.ndim() == 4) {
    return {slice4, slice3, slice2, slice1};
  } else {
    throw std::runtime_error("part: ndim must be 3 or 4.");
  }
}

double MeshBlockImpl::initialize(Variables& vars) {
  /*c10d::BarrierOptions op;
  op.device_ids = {options->layout()->local_rank()};
  _playout->pg->barrier(op)->wait();*/
  _playout->pg->barrier()->wait();

  //// ------------ (1) Set up a signal handler ------------ ////
  SignalHandler::GetInstance();

  if (pintg->options->restart() != "") {
    return _init_from_restart(vars);
  }

  BoundaryFuncOptions bops;
  bops.nghost(options->coord()->nghost());

  torch::Tensor hydro_w, scalar_r, solid;

  //// ------------ (2) Check hydro primitive ------------ ////
  int64_t nc3 = options->coord()->nc3();
  int64_t nc2 = options->coord()->nc2();
  int64_t nc1 = options->coord()->nc1();

  TORCH_CHECK(vars.count("hydro_w"),
              "initialize: hydro_w is required for hydro model.");
  hydro_w = vars.at("hydro_w");

  TORCH_CHECK(hydro_w.sizes() ==
                  std::vector<int64_t>({phydro->peos->nvar(), nc3, nc2, nc1}),
              "initialize: hydro_w has incorrect shape.", " Expected [",
              phydro->peos->nvar(), ", ", nc3, ", ", nc2, ", ", nc1,
              "] but got ", hydro_w.sizes());

  //// -------- (3) Apply hydro primitive boundary condition -------- ////
  if (options->verbose()) {
    SINFO(MeshBlock) << "applying hydro primitive boundary conditions."
                     << std::endl;
  }

  bops.type(kPrimitive);
  for (int i = 0; i < options->bfuncs().size(); ++i) {
    if (options->bfuncs()[i] == nullptr) continue;
    options->bfuncs()[i](vars.at("hydro_w"), 3 - i / 2, bops);
  }

  //// ----------- (4) Check scalar primitive ---------- ////
  if (pscalar->nvar() > 0) {
    TORCH_CHECK(vars.count("scalar_r"),
                "initialize: scalar_r is required for scalar model.");
    scalar_r = vars.at("scalar_r");
    TORCH_CHECK(scalar_r.sizes() ==
                    std::vector<int64_t>({pscalar->nvar(), nc3, nc2, nc1}),
                "initialize: scalar_r has incorrect shape.", " Expected [",
                pscalar->nvar(), ", ", nc3, ", ", nc2, ", ", nc1, "] but got ",
                scalar_r.sizes());

    //// ------- (5) Apply scalar primitive boundary condition -------- ////
    if (options->verbose()) {
      SINFO(MeshBlock) << "applying scalar primitive boundary conditions."
                       << std::endl;
    }

    bops.type(kScalar);
    for (int i = 0; i < options->bfuncs().size(); ++i) {
      if (options->bfuncs()[i] == nullptr) continue;
      options->bfuncs()[i](vars.at("scalar_r"), 3 - i / 2, bops);
    }
  }

  //// ------ (6) Exchange hydro and scalar buffers -------- ////
  if (options->verbose()) {
    SINFO(MeshBlock) << "exchanging ghost zones." << std::endl;
  }

  SyncOptions sync_opts;
  sync_opts.interpolate(true).type(kPrimitive);

  Variables sync_vars;
  sync_vars["hydro_w"] = hydro_w;

  std::vector<c10::intrusive_ptr<c10d::Work>> works;
  _playout->forward(this, sync_vars, sync_opts, works);
  _playout->finalize(this, sync_vars, sync_opts, works);

  if (pscalar->nvar() > 0) {
    sync_opts.type(kScalar);
    sync_vars.clear();
    sync_vars["scalar_r"] = scalar_r;
    _playout->forward(this, sync_vars, sync_opts, works);
    _playout->finalize(this, sync_vars, sync_opts, works);
  }

  //// ------ (7) Computer hydro and scalar conserved -------- ////
  if (options->verbose()) {
    SINFO(MeshBlock) << "computing conserved variables." << std::endl;
  }

  vars["hydro_u"] = phydro->peos->compute("W->U", {hydro_w});
  if (pscalar->nvar() > 0) {
    vars["scalar_s"] = hydro_w[IDN] * scalar_r;
  }

  //// ------------- (8) Fill solid boundaries -------------- ////
  if (vars.count("solid")) {
    if (options->verbose()) {
      SINFO(MeshBlock) << "filling solid boundaries." << std::endl;
    }

    solid = vars.at("solid");
    TORCH_CHECK(solid.sizes() == std::vector<int64_t>({nc3, nc2, nc1}),
                "initialize: solid has incorrect shape.", " Expected [", nc3,
                ", ", nc2, ", ", nc1, "] but got ", solid.sizes());
    vars["fill_solid_hydro_w"] =
        torch::where(solid.unsqueeze(0).expand_as(hydro_w), hydro_w, 0.);
    vars["fill_solid_hydro_w"].narrow(0, IVX, 3).zero_();
    pib->mark_prim_solid_(hydro_w, solid);

    vars["fill_solid_hydro_u"] =
        torch::where(solid.unsqueeze(0).expand_as(vars.at("hydro_u")),
                     vars.at("hydro_u"), 0.);
    vars["fill_solid_hydro_u"].narrow(0, IVX, 3).zero_();
  } else {
    vars["fill_solid_hydro_w"] = hydro_w;
    vars["fill_solid_hydro_u"] = vars.at("hydro_u");
  }

  //// -------- (9) Apply hydro conservd boundary condition -------- ////
  if (options->verbose()) {
    SINFO(MeshBlock) << "applying hydro conserved boundary conditions."
                     << std::endl;
  }

  bops.type(kConserved);
  for (int i = 0; i < options->bfuncs().size(); ++i) {
    if (options->bfuncs()[i] == nullptr) continue;
    options->bfuncs()[i](vars.at("hydro_u"), 3 - i / 2, bops);
  }

  //// ------- (10) Apply scalar conserved boundary condition -------- ////
  if (pscalar->nvar() > 0) {
    if (options->verbose()) {
      SINFO(MeshBlock) << "applying scalar conserved boundary conditions."
                       << std::endl;
    }

    bops.type(kScalar);
    for (int i = 0; i < options->bfuncs().size(); ++i) {
      if (options->bfuncs()[i] == nullptr) continue;
      options->bfuncs()[i](vars.at("scalar_s"), 3 - i / 2, bops);
    }
  }

  //// ---------------- (11) Start timing ----------------- ////
  _time_start = clock();

  if (options->verbose()) {
    SINFO(MeshBlock) << "initialization completed." << std::endl;
  }

  return 0.;  // default start time is 0.0
}

double MeshBlockImpl::max_time_step(Variables const& vars) {
  auto const& w = vars.at("hydro_w");
  auto dt_min =
      torch::tensor({1.e9}, torch::dtype(torch::kFloat64).device(w.device()));

  // hyperbolic hydro time step
  if (vars.count("solid")) {
    dt_min[0] = phydro->max_time_step(w, vars.at("solid"));
  } else {
    dt_min[0] = phydro->max_time_step(w);
  }

  // gather the minimum dt across all ranks
  std::vector<at::Tensor> dt_reduce = {dt_min};

  c10d::AllreduceOptions op;
  op.reduceOp = c10d::ReduceOp::MIN;
  _playout->pg->allreduce(dt_reduce, op)->wait();

  auto dt = dt_reduce[0].item<double>();

  if (options->verbose()) {
    SINFO(MeshBlock) << "suggested dt from hydro: " << std::scientific
                     << std::setprecision(6) << dt << std::endl;
  }
  return pow(2., -pintg->current_redo) * pintg->options->cfl() * dt;
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
  // (3.A) hydro forward
  fut_hydro_du = phydro->forward(dt, hydro_u, vars);
  if (options->verbose()) {
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    SINFO(MeshBlock) << "stage " << stage
                     << " hydro forward time (s): " << elapsed.count()
                     << std::endl;
    start = std::chrono::high_resolution_clock::now();
  }

  // (3.B) scalar forward
  if (pscalar->nvar() > 0) {
    fut_scalar_ds = pscalar->forward(dt, scalar_s, vars);
    if (options->verbose()) {
      auto end = std::chrono::high_resolution_clock::now();
      std::chrono::duration<double> elapsed = end - start;
      SINFO(MeshBlock) << "stage " << stage
                       << " scalar forward time (s): " << elapsed.count()
                       << std::endl;
      start = std::chrono::high_resolution_clock::now();
    }
  }

  // -------- (4) multi-stage averaging --------
  hydro_u.set_(pintg->forward(stage, _hydro_u0, _hydro_u1, fut_hydro_du));
  phydro->peos->apply_conserved_limiter_(hydro_u);

  if (options->verbose()) {
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    SINFO(MeshBlock) << "stage " << stage
                     << " multi-stage averaging time (s): " << elapsed.count()
                     << std::endl;
    start = std::chrono::high_resolution_clock::now();
  }

  if (pscalar->nvar() > 0) {
    scalar_s.set_(pintg->forward(stage, _scalar_s0, _scalar_s1, fut_scalar_ds));
    if (options->verbose()) {
      auto end = std::chrono::high_resolution_clock::now();
      std::chrono::duration<double> elapsed = end - start;
      SINFO(MeshBlock) << "stage " << stage
                       << " multi-stage scalar averaging time (s): "
                       << elapsed.count() << std::endl;
      start = std::chrono::high_resolution_clock::now();
    }
  }

  // -------- (5) update ghost zones --------
  BoundaryFuncOptions bops;
  bops.nghost(options->coord()->nghost());

  if (vars.count("solid")) {
    pib->fill_cons_solid_(hydro_u, vars.at("solid"),
                          vars.at("fill_solid_hydro_u"));
    if (options->verbose()) {
      auto end = std::chrono::high_resolution_clock::now();
      std::chrono::duration<double> elapsed = end - start;
      SINFO(MeshBlock) << "stage " << stage
                       << " fill solid hydro conserved time (s): "
                       << elapsed.count() << std::endl;
      start = std::chrono::high_resolution_clock::now();
    }
  }

  // (5.A) apply hydro boundary
  bops.type(kConserved);
  for (int i = 0; i < options->bfuncs().size(); ++i) {
    if (options->bfuncs()[i] == nullptr) continue;
    options->bfuncs()[i](hydro_u, 3 - i / 2, bops);
  }
  if (options->verbose()) {
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    SINFO(MeshBlock) << "stage " << stage
                     << " hydro boundary condition time (s): "
                     << elapsed.count() << std::endl;
    start = std::chrono::high_resolution_clock::now();
  }

  // (5.B) apply scalar boundary
  if (pscalar->nvar() > 0) {
    bops.type(kScalar);
    for (int i = 0; i < options->bfuncs().size(); ++i) {
      if (options->bfuncs()[i] == nullptr) continue;
      options->bfuncs()[i](scalar_s, 3 - i / 2, bops);
    }
    if (options->verbose()) {
      auto end = std::chrono::high_resolution_clock::now();
      std::chrono::duration<double> elapsed = end - start;
      SINFO(MeshBlock) << "stage " << stage
                       << " scalar boundary condition time (s): "
                       << elapsed.count() << std::endl;
      start = std::chrono::high_resolution_clock::now();
    }
  }

  // -------- (6) saturation adjustment --------
  if (stage == pintg->stages.size() - 1 && phydro->options->eos()->thermo() &&
      phydro->options->eos()->thermo()->reactions().size() > 0) {
    phydro->peos->apply_conserved_limiter_(hydro_u);

    int ny = hydro_u.size(0) - 5;  // number of species

    auto ke = phydro->peos->compute("U->K", {hydro_u});
    auto rho = hydro_u[IDN] + hydro_u.narrow(0, ICY, ny).sum(0);
    auto ie = hydro_u[IPR] - ke;

    auto yfrac = hydro_u.narrow(0, ICY, ny) / rho;

    auto m = named_modules()["hydro.eos.thermo"];
    auto pthermo = std::dynamic_pointer_cast<kintera::ThermoYImpl>(m);

    auto sub = part({0, 0, 0}, PartOptions().exterior(false));
    auto sub3 = part({0, 0, 0}, PartOptions().exterior(false).ndim(3));
    pthermo->forward(rho.index(sub3), ie.index(sub3), yfrac.index(sub),
                     /*warm_start=*/true);

    hydro_u.narrow(0, ICY, ny) = yfrac * rho;
    if (options->verbose()) {
      auto end = std::chrono::high_resolution_clock::now();
      std::chrono::duration<double> elapsed = end - start;
      SINFO(MeshBlock) << "stage " << stage
                       << " saturation adjustment time (s): " << elapsed.count()
                       << std::endl;
      start = std::chrono::high_resolution_clock::now();
    }
  }

  // -------- (7) ghost zone exchange --------
  SyncOptions sync_opts;
  sync_opts.interpolate(true).type(kConserved);

  Variables sync_vars;
  sync_vars["hydro_u"] = hydro_u;

  std::vector<c10::intrusive_ptr<c10d::Work>> works;
  _playout->forward(this, sync_vars, sync_opts, works);
  _playout->finalize(this, sync_vars, sync_opts, works);

  if (pscalar->nvar() > 0) {
    sync_opts.type(kScalar);
    sync_vars.clear();
    sync_vars["scalar_s"] = scalar_s;
    _playout->forward(this, sync_vars, sync_opts, works);
    _playout->finalize(this, sync_vars, sync_opts, works);
  }

  if (options->verbose()) {
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    SINFO(MeshBlock) << "stage " << stage
                     << " ghost zone exchange time (s): " << elapsed.count()
                     << std::endl;
    start = std::chrono::high_resolution_clock::now();
  }

  // -------- (8) save final state for next stage --------
  _hydro_u1.copy_(hydro_u);
  if (pscalar->nvar() > 0) {
    _scalar_s1.copy_(scalar_s);
  }
}

void MeshBlockImpl::make_outputs(Variables const& vars, double current_time,
                                 bool final_write) {
  for (auto& output_type : output_types) {
    if (final_write) {
      output_type->write_output_file(this, vars, current_time, final_write);
    } else if (current_time >= output_type->next_time) {
      output_type->write_output_file(this, vars, current_time, final_write);
      output_type->next_time += output_type->options->dt();
      output_type->file_number += 1;
    }
  }
  if (options->verbose()) {
    SINFO(MeshBlock) << "output writing completed at time: " << current_time
                     << std::endl;
  }
}

void MeshBlockImpl::print_cycle_info(Variables const& vars, double time,
                                     double dt) const {
  const int dt_precision = std::numeric_limits<double>::max_digits10 - 3;
  bool compute_mass = false;
  bool compute_energy = false;

  c10d::ReduceOptions opsum;
  opsum.reduceOp = c10d::ReduceOp::SUM;
  opsum.rootRank = options->layout()->root_rank();

  if (pintg->options->ncycle_out() != 0) {
    if (cycle % pintg->options->ncycle_out() == 0) {
      if (vars.count("hydro_u")) {
        compute_mass = true;
        compute_energy = phydro->peos->nvar() > IPR;
      }

      SINFO() << "cycle=" << cycle << " redo=" << pintg->current_redo
              << std::scientific << std::setprecision(dt_precision)
              << " time=" << time << " dt=" << dt;

      auto interior = part({0, 0, 0}, PartOptions().exterior(false));

      auto vol = pcoord->cell_volume();
      auto hydro_u_tol = vars.at("hydro_u") * vol;

      std::vector<at::Tensor> sum = {
          hydro_u_tol.index(interior).sum({1, 2, 3})};
      _playout->pg->reduce(sum, opsum)->wait();

      if (compute_mass) {
        auto mass = sum[0][IDN];
        SINFO() << std::scientific << std::setprecision(dt_precision)
                << " mass0=" << mass.item<double>();

        int ny = hydro_u_tol.size(0) - 5;  // number of species
        if (ny > 0) {
          for (int n = 0; n < ny; ++n) {
            mass += sum[0][ICY + n];
          }
          SINFO() << std::scientific << std::setprecision(dt_precision)
                  << " masst=" << mass.item<double>();
        }
      }

      if (compute_energy) {
        SINFO() << std::scientific << std::setprecision(dt_precision)
                << " energy=" << sum[0][IPR].item<double>();
      }

      SINFO() << std::endl;
    }
  }
}

void MeshBlockImpl::finalize(Variables const& vars, double time) {
  // make final output
  make_outputs(vars, time, /*final_write=*/true);

  auto sig = SignalHandler::GetInstance();
  if (sig->GetSignalFlag(SIGTERM) != 0) {
    SINFO() << std::endl << "Terminating on Terminate signal" << std::endl;
  } else if (sig->GetSignalFlag(SIGINT) != 0) {
    SINFO() << std::endl << "Terminating on Interrupt signal" << std::endl;
  } else if (sig->GetSignalFlag(SIGALRM) != 0) {
    SINFO() << std::endl << "Terminating on wall-time limit" << std::endl;
  } else if (pintg->options->nlim() >= 0 && cycle >= pintg->options->nlim()) {
    SINFO() << std::endl << "Terminating on cycle limit" << std::endl;
  } else if (time >= pintg->options->tlim()) {
    SINFO() << std::endl << "Terminating on time limit" << std::endl;
  } else {
    SINFO() << std::endl << "Terminating abnormally" << std::endl;
  }

  SINFO() << "time=" << time << " cycle=" << cycle - 1 << std::endl;
  SINFO() << "tlim=" << pintg->options->tlim()
          << " nlim=" << pintg->options->nlim() << std::endl;

  // ---------- timing info ----------
  clock_t tstop = clock();
  double cpu_time =
      (tstop > _time_start ? static_cast<double>(tstop - _time_start) : 1.0) /
      static_cast<double>(CLOCKS_PER_SEC);

  std::vector<at::Tensor> cells = {
      torch::tensor({_hydro_u0.size(1) * _hydro_u0.size(2) * _hydro_u0.size(3)},
                    torch::dtype(torch::kInt64).device(device()))};

  c10d::ReduceOptions opsum;
  opsum.reduceOp = c10d::ReduceOp::SUM;
  opsum.rootRank = options->layout()->root_rank();

  _playout->pg->reduce(cells, opsum)->wait();

  int64_t cellcycles = cells[0].item<int64_t>() * cycle * pintg->stages.size();
  double zc_cpus = static_cast<double>(cellcycles) / cpu_time;

  SINFO() << std::endl
          << "million cells-per-cycle = " << cellcycles / 1e6 << std::endl;
  SINFO() << "cpu time used (s) = " << cpu_time << std::endl;
  SINFO() << "million cell-updates/second = " << zc_cpus / 1e6 << std::endl;

  // ------ shutdown processing group ------
  /*c10d::BarrierOptions op;
  op.device_ids = {options->layout()->local_rank()};
  _playout->pg->barrier(op)->wait();*/
  _playout->pg->barrier()->wait();

  _playout->send_bufs.clear();
  _playout->send_bufs.shrink_to_fit();

  _playout->recv_bufs.clear();
  _playout->recv_bufs.shrink_to_fit();

  _playout->pg->shutdown();
}

int MeshBlockImpl::check_redo(Variables& vars) {
  auto sig = snap::SignalHandler::GetInstance();
  if (sig->CheckSignalFlags(this)) return -1;  // terminate

  // check if density or pressure is negative
  auto hydro_u = vars.at("hydro_u");
  auto interior = part({0, 0, 0}, PartOptions().exterior(false));
  auto redo_rho = hydro_u.index(interior)[IDN].min().item<double>() <= 0.;
  auto redo_pres = hydro_u.size(0) > IPR &&
                   hydro_u.index(interior)[IPR].min().item<double>() <= 0.;

  if (redo_rho || redo_pres) {
    SINFO(MeshBlock)
        << "Negative density/pressure detected. Redoing the step with "
           "smaller dt."
        << std::endl;
    pintg->current_redo += 1;
    if (pintg->current_redo > pintg->options->max_redo()) {
      SINFO(MeshBlock)
          << "Maximum number of redo attempts exceeded. Terminating."
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

torch::Device MeshBlockImpl::device() const {
  if (_playout->pg->getBoundDeviceId().has_value()) {
    return _playout->pg->getBoundDeviceId().value();
  } else {
    return torch::Device("cpu");
  }
}

double MeshBlockImpl::_init_from_restart(Variables& vars) {
  // create filename: <file_basename>.<block_id>.<fid>.restart
  std::string fid = pintg->options->restart();

  std::string fname;
  char bid[12];
  snprintf(bid, sizeof(bid), "block%d", options->layout()->rank());

  fname.append(options->basename());
  fname.append(".");
  fname.append(bid);
  fname.append(".");
  fname.append(fid);
  fname.append(".restart");

  // load variables from disk
  kintera::load_tensors(vars, fname);

  // load auxiliary timing data from disk
  Variables timing_vars;

  timing_vars["last_time"] = torch::Tensor();
  timing_vars["last_cycle"] = torch::Tensor();
  timing_vars["file_number"] = torch::Tensor();
  timing_vars["next_time"] = torch::Tensor();

  kintera::load_tensors(timing_vars, fname);

  cycle = timing_vars.at("last_cycle").item<int64_t>();
  for (int n = 0; n < output_types.size(); ++n) {
    output_types[n]->file_number =
        timing_vars.at("file_number")[n].item<int64_t>();
    output_types[n]->next_time = timing_vars.at("next_time")[n].item<double>();
  }

  // start timing
  _time_start = clock();
  _cycle_start = cycle;

  return timing_vars.at("last_time").item<double>();
}

}  // namespace snap
