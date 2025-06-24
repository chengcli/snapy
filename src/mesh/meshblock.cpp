// base
#include <configure.h>

// snap
#include <snap/input/parameter_input.hpp>

// snap
#include "mesh_formatter.hpp"
#include "meshblock.hpp"

namespace snap {

MeshBlockImpl::MeshBlockImpl(MeshBlockOptions const& options_)
    : options(std::move(options_)) {
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

  // set up integrator
  pintg = register_module("intg", Integrator(options.intg()));

  // set up hydro model
  phydro = register_module("hydro", Hydro(options.hydro()));
  options.hydro() = phydro->options;

  // set up scalar model
  pscalar = register_module("scalar", Scalar(options.scalar()));

  // set up data
  hydro_u0_ = register_buffer(
      "HU0", torch::zeros({phydro->peos->nvar(), nc3(), nc2(), nc1()},
                          torch::kFloat64));

  hydro_u1_ = register_buffer(
      "HU1", torch::zeros({phydro->peos->nvar(), nc3(), nc2(), nc1()},
                          torch::kFloat64));

  if (pscalar->nvar() > 0) {
    scalar_u = register_buffer(
        "scalar_u",
        torch::zeros({pscalar->nvar(), nc3(), nc2(), nc1()}, torch::kFloat64));
  }
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

void MeshBlockImpl::set_primitives(torch::Tensor const& hydro_w,
                                   torch::optional<torch::Tensor> scalar_w) {
  auto hydro_u = phydro->peos->compute("W->U", {hydro_w});

  if (pscalar->nvar() > 0) {
    // scalar_u.set_(scalar_w);
  }

  BoundaryFuncOptions op;
  op.nghost(options.nghost());
  op.type(kConserved);
  for (int i = 0; i < bfuncs.size(); ++i) bfuncs[i](hydro_u, 3 - i / 2, op);

  phydro->peos->forward(hydro_u, hydro_w);
}

double MeshBlockImpl::max_root_time_step(int root_level,
                                         torch::optional<torch::Tensor> solid) {
  double dt = 1.e9;
  auto const& w = phydro->peos->get_buffer("W");

  if (phydro->peos->nvar() > 0) {
    dt = std::min(dt, phydro->max_time_step(w, solid));
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

  auto const& hydro_u = phydro->peos->get_buffer("U");

  // -------- (1) save initial state --------
  if (stage == 0) {
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
  if (phydro->peos->nvar() > 0) {
    fut_hydro_du = phydro->forward(hydro_u, dt, solid);
  }

  // (3.2) scalar forward
  if (pscalar->nvar() > 0) {
    fut_scalar_du = pscalar->forward(scalar_u, dt);
  }

  // -------- (4) multi-stage averaging --------
  if (phydro->peos->nvar() > 0) {
    hydro_u.set_(pintg->forward(stage, hydro_u0_, hydro_u1_, fut_hydro_du));
    hydro_u1_.copy_(hydro_u);
  }

  if (pscalar->nvar() > 0) {
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
