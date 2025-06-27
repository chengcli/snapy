// snap
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
  // set up integrator
  pintg = register_module("intg", Integrator(options.intg()));
  options.intg() = pintg->options;

  // set up hydro model
  phydro = register_module("hydro", Hydro(options.hydro()));
  options.hydro() = phydro->options;

  // set up scalar model
  pscalar = register_module("scalar", Scalar(options.scalar()));
  options.scalar() = pscalar->options;

  // set up hydro buffer
  auto const& hydro_u = phydro->peos->get_buffer("U");
  _hydro_u0 = register_buffer("HU0", torch::zeros_like(hydro_u));
  _hydro_u1 = register_buffer("HU1", torch::zeros_like(hydro_u));

  // set up scalar buffer
  auto const& scalar_v = pscalar->get_buffer("V");
  _scalar_v0 = register_buffer("SV0", torch::zeros_like(scalar_v));
  _scalar_v1 = register_buffer("SV0", torch::zeros_like(scalar_v));
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

void MeshBlockImpl::initialize(torch::Tensor const& hydro_w,
                               torch::Tensor const& scalar_x) {
  BoundaryFuncOptions op;
  op.nghost(options.hydro().coord().nghost());

  // hydro
  if (phydro->peos->nvar() > 0) {
    auto const& hydro_u = phydro->peos->compute("W->U", {hydro_w});

    op.type(kConserved);
    for (int i = 0; i < options.bfuncs().size(); ++i) {
      options.bfuncs()[i](hydro_u, 3 - i / 2, op);
    }

    phydro->peos->forward(hydro_u, /*out=*/hydro_w);
  }

  // scalar
  if (pscalar->nvar() > 0) {
    auto const& temp = phydro->peos->get_buffer("thermo.T");
    auto const& scalar_v = pscalar->pthermo->compute(
        "TPX->V", {temp, hydro_w[Index::IPR], scalar_x});

    op.type(kScalar);
    for (int i = 0; i < options.bfuncs().size(); ++i) {
      options.bfuncs()[i](scalar_v, 3 - i / 2, op);
    }

    // FIXME: scalar should have an eos as well
    // scalar_x.set_(pscalar->pthermo->compute("V->X", {scalar_v}));
  }
}

double MeshBlockImpl::max_time_step(torch::Tensor solid) {
  double dt = 1.e9;
  auto const& w = phydro->peos->get_buffer("W");
  auto const& x = pscalar->get_buffer("X");

  if (phydro->peos->nvar() > 0) {
    dt = std::min(dt, phydro->max_time_step(w, solid));
  }

  if (pscalar->nvar() > 0) {
    dt = std::min(dt, pscalar->max_time_step(x));
  }

  return pintg->options.cfl() * dt;
}

int MeshBlockImpl::forward(double dt, int stage, torch::Tensor solid) {
  TORCH_CHECK(stage >= 0 && stage < pintg->stages.size(),
              "Invalid stage: ", stage);

  auto const& hydro_u = phydro->peos->get_buffer("U");
  auto const& scalar_v = pscalar->get_buffer("V");

  // -------- (1) save initial state --------
  if (stage == 0) {
    if (phydro->peos->nvar() > 0) {
      _hydro_u0.copy_(hydro_u);
      _hydro_u1.copy_(hydro_u);
    }

    if (pscalar->nvar() > 0) {
      _scalar_v0.copy_(scalar_v);
      _scalar_v1.copy_(scalar_v);
    }
  }

  // -------- (2) set containers for future results --------
  torch::Tensor fut_hydro_du, fut_scalar_dv;

  // -------- (3) launch all jobs --------
  // (3.1) hydro forward
  if (phydro->peos->nvar() > 0) {
    fut_hydro_du = phydro->forward(hydro_u, dt, solid);
  }

  // (3.2) scalar forward
  if (pscalar->nvar() > 0) {
    fut_scalar_dv = pscalar->forward(scalar_v, dt);
  }

  // -------- (4) multi-stage averaging --------
  if (phydro->peos->nvar() > 0) {
    hydro_u.set_(pintg->forward(stage, _hydro_u0, _hydro_u1, fut_hydro_du));
    _hydro_u1.copy_(hydro_u);
  }

  if (pscalar->nvar() > 0) {
    scalar_v.set_(pintg->forward(stage, _scalar_v0, _scalar_v1, fut_scalar_dv));
    _scalar_v1.copy_(scalar_v);
  }

  // -------- (5) update ghost zones --------
  BoundaryFuncOptions op;
  op.nghost(options.hydro().coord().nghost());

  // (5.1) apply hydro boundary
  if (phydro->peos->nvar() > 0) {
    op.type(kConserved);
    for (int i = 0; i < options.bfuncs().size(); ++i)
      options.bfuncs()[i](hydro_u, 3 - i / 2, op);
  }

  // (5.2) apply scalar boundary
  if (pscalar->nvar() > 0) {
    op.type(kScalar);
    for (int i = 0; i < options.bfuncs().size(); ++i)
      options.bfuncs()[i](scalar_v, 3 - i / 2, op);
  }

  return 0;
}

}  // namespace snap
