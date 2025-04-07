// spdlog
#include <configure.h>
#include <spdlog/sinks/basic_file_sink.h>

// base
#include <globals.h>

#include <formatter.hpp>
#include <input/parameter_input.hpp>

// snap
#include "mesh_formatter.hpp"
#include "meshblock.hpp"

namespace snap {
MeshBlockOptions::MeshBlockOptions(ParameterInput pin) {
  nghost(pin->GetOrAddInteger("meshblock", "nghost", 2));

  hydro(HydroOptions(pin));
  scalar(ScalarOptions(pin));
}

MeshBlockImpl::MeshBlockImpl(MeshBlockOptions const& options_)
    : options(options_) {
  // set up logical location
  options.loc(torch::nn::AnyModule(LogicalLocation()));

  if (nc1() > 1 && options.bflags().size() < 2) {
    throw std::runtime_error("MeshBlockImpl: bflags size must be at least 2");
  }

  if (nc2() > 1 && options.bflags().size() < 4) {
    throw std::runtime_error("MeshBlockImpl: bflags size must be at least 4");
  }

  if (nc3() > 1 && options.bflags().size() < 6) {
    throw std::runtime_error("MeshBlockImpl: bflags size must be at least 6");
  }

  bfuncs.resize(options.bflags().size());
  for (int i = 0; i < options.bflags().size(); ++i) {
    bfuncs[i] = get_boundary_function(static_cast<BoundaryFace>(i),
                                      options.bflags()[i]);
  }
  reset();
}

MeshBlockImpl::MeshBlockImpl(MeshBlockOptions const& options_,
                             LogicalLocation ploc_)
    : options(options_) {
  // set up logical location
  options.loc(torch::nn::AnyModule(ploc_));
  reset();
}

void MeshBlockImpl::reset() {
  if (options.loc().is_empty()) {
    ploc = register_module("loc", LogicalLocation());
  } else {
    ploc = register_module("loc", options.loc().clone().get<LogicalLocation>());
  }
  LOG_INFO(logger, "{} is at logical location: {}", name(), ploc);

  // set up integrator
  pintg = register_module("intg", Integrator(options.intg()));

  // set up hydro model
  options.hydro().nghost(options.nghost());
  phydro = register_module("hydro", Hydro(options.hydro()));
  options.hydro() = phydro->options;

  // set up scalar model
  pscalar = register_module("scalar", Scalar(options.scalar()));

  // set up data
  hydro_u = register_buffer(
      "hydro_u",
      torch::zeros({phydro->nvar(), nc3(), nc2(), nc1()}, torch::kFloat64));
  LOG_INFO(logger, "{} registers hydro_u with shape: {}", name(),
           hydro_u.sizes());

  hydro_u0_ = register_buffer(
      "hydro_u0",
      torch::zeros({phydro->nvar(), nc3(), nc2(), nc1()}, torch::kFloat64));
  LOG_INFO(logger, "{} registers hydro_u0 with shape: {}", name(),
           hydro_u0_.sizes());

  hydro_u1_ = register_buffer(
      "hydro_u1",
      torch::zeros({phydro->nvar(), nc3(), nc2(), nc1()}, torch::kFloat64));
  LOG_INFO(logger, "{} registers hydro_u1 with shape: {}", name(),
           hydro_u1_.sizes());

  if (pscalar->nvar() > 0) {
    scalar_u = register_buffer(
        "scalar_u",
        torch::zeros({pscalar->nvar(), nc3(), nc2(), nc1()}, torch::kFloat64));
    LOG_INFO(logger, "{} reigsters scalar_u with shape: {}", name(),
             scalar_u.sizes());
  }

  LOG_INFO(logger, "{} resets with options: {}", name(), options);
}

std::vector<torch::indexing::TensorIndex> MeshBlockImpl::part(
    std::tuple<int, int, int> offset, bool exterior, int extend_x1,
    int extend_x2, int extend_x3) const {
  int is_ghost = exterior ? 1 : 0;

  auto [o3, o2, o1] = offset;
  int start1, len1, start2, len2, start3, len3;

  int nx1 = nc1() > 1 ? nc1() - 2 * options.nghost() : 1;
  int nx2 = nc2() > 1 ? nc2() - 2 * options.nghost() : 1;
  int nx3 = nc3() > 1 ? nc3() - 2 * options.nghost() : 1;

  // ---- dimension 1 ---- //
  int nghost = nx1 == 1 ? 0 : options.nghost();

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
  nghost = nx2 == 1 ? 0 : options.nghost();

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
  nghost = nx3 == 1 ? 0 : options.nghost();

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

void MeshBlockImpl::set_primitives(torch::Tensor hydro_w,
                                   torch::optional<torch::Tensor> scalar_w) {
  LOG_INFO(logger, "{} sets primitives", name());
  phydro->peos->prim2cons(hydro_u, hydro_w);

  if (pscalar->nvar() > 0) {
    // scalar_u.set_(scalar_w);
  }

  BoundaryFuncOptions op;
  op.nghost(options.nghost());
  op.type(kConserved);
  for (int i = 0; i < bfuncs.size(); ++i) bfuncs[i](hydro_u, 3 - i / 2, op);

  SET_SHARED("hydro/w") = phydro->peos->forward(hydro_u);
}

double MeshBlockImpl::max_root_time_step(int root_level,
                                         torch::optional<torch::Tensor> solid) {
  LOG_INFO(logger, "{} {} projects time step to root_level = {}", name(), ploc,
           root_level);

  double dt = 1.e9;
  if (!HAS_SHARED("hydro/w")) {
    SET_SHARED("hydro/w") = phydro->peos->forward(hydro_u);
  }

  if (phydro->nvar() > 0) {
    dt = std::min(dt, phydro->max_time_step(GET_SHARED("hydro/w"), solid));
  }

  if (pscalar->nvar() > 0) {
    dt = std::min(dt, pscalar->max_time_step(scalar_u));
  }

  return pintg->options.cfl() * dt * (1 << (ploc->level - root_level));
}

int MeshBlockImpl::forward(double dt, int stage,
                           torch::optional<torch::Tensor> solid) {
  TORCH_CHECK(stage >= 0 && stage < pintg->stages.size(),
              "Invalid stage: ", stage);

  LOG_INFO(logger, "{} marches with dt = {}", name(), dt);
  torch::NoGradGuard no_grad;

  // -------- (1) save initial state --------
  if (stage == 0) {
    LOG_INFO(logger, "{} {} saves initial state", name(), ploc);
    hydro_u0_.copy_(hydro_u);
    hydro_u1_.copy_(hydro_u);
    if (pscalar->nvar() > 0) {
      scalar_u0_.copy_(scalar_u);
      scalar_u1_.copy_(scalar_u);
    }
  }

  // -------- (2) set containers for future results --------
  torch::Tensor fut_hydro_du, fut_scalar_du;

  // -------- (3) launch all jobs --------
  // (3.1) hydro forward
  if (phydro->nvar() > 0) {
    LOG_INFO(logger, "Stage-{}, {} {} advances hydro", stage + 1, name(), ploc);
    fut_hydro_du = phydro->forward(hydro_u, dt, solid);
  }

  // (3.2) scalar forward
  if (pscalar->nvar() > 0) {
    LOG_INFO(logger, "Stage-{}, {} {} advances scalar", stage + 1, name(),
             ploc);
    fut_scalar_du = pscalar->forward(scalar_u, dt);
  }

  // -------- (4) multi-stage averaging --------
  if (phydro->nvar() > 0) {
    LOG_INFO(logger, "Stage-{}, {} {} averages hydro", stage + 1, name(), ploc);
    hydro_u.copy_(pintg->forward(stage, hydro_u0_, hydro_u1_, fut_hydro_du));
    hydro_u1_.copy_(hydro_u);
  }

  if (pscalar->nvar() > 0) {
    LOG_INFO(logger, "Stage-{], {} {} averages scalar", stage + 1, name(),
             ploc);
    scalar_u.copy_(
        pintg->forward(stage, scalar_u0_, scalar_u1_, fut_scalar_du));
    scalar_u1_.copy_(scalar_u);
  }

  // -------- (5) exchange boundary and update ghost zones --------
  BoundaryFuncOptions op;
  op.nghost(options.nghost());
  op.type(kConserved);
  for (int i = 0; i < bfuncs.size(); ++i) bfuncs[i](hydro_u, 3 - i / 2, op);

  return 0;
}
}  // namespace snap
