// fmt
#include <spdlog/spdlog.h>

// base
#include <globals.h>

// base
#include <input/parameter_input.hpp>

// fvm
#include <fvm/index.h>

#include <fvm/hydro/hydro.hpp>

#include "output.hpp"

namespace snap {
void OutputType::loadHydroOutputData(MeshBlock pmb) {
  OutputData *pod;
  auto peos = pmb->phydro->peos;

  // (lab-frame) density
  if (ContainVariable(options.variable(), "D") ||
      ContainVariable(options.variable(), "cons")) {
    pod = new OutputData;
    pod->type = "SCALARS";
    pod->name = "dens";
    pod->data.InitFromTensor(pmb->hydro_u, 4, index::IDN, 1);
    AppendOutputDataNode(pod);
    num_vars_++;
  }

  // (rest-frame) density
  if (ContainVariable(options.variable(), "d") ||
      ContainVariable(options.variable(), "prim")) {
    pod = new OutputData;
    pod->type = "SCALARS";
    pod->name = "rho";
    pod->data.InitFromTensor(GET_SHARED("hydro/w"), 4, index::IDN, 1);
    AppendOutputDataNode(pod);
    num_vars_++;
  }

  // total energy
  if (peos->nhydro() > 4) {
    if (ContainVariable(options.variable(), "E") ||
        ContainVariable(options.variable(), "cons")) {
      pod = new OutputData;
      pod->type = "SCALARS";
      pod->name = "Etot";
      pod->data.InitFromTensor(pmb->hydro_u, 4, index::IPR, 1);

      AppendOutputDataNode(pod);
      num_vars_++;
    }

    // pressure
    if (ContainVariable(options.variable(), "p") ||
        ContainVariable(options.variable(), "prim")) {
      pod = new OutputData;
      pod->type = "SCALARS";
      pod->name = "press";
      pod->data.InitFromTensor(GET_SHARED("hydro/w"), 4, index::IPR, 1);
      AppendOutputDataNode(pod);
      num_vars_++;
    }
  }

  // momentum vector
  if (ContainVariable(options.variable(), "m") ||
      ContainVariable(options.variable(), "cons")) {
    pod = new OutputData;
    pod->type = "VECTORS";
    pod->name = "mom";
    pod->data.InitFromTensor(pmb->hydro_u, 4, index::IVX, 3);

    AppendOutputDataNode(pod);
    num_vars_ += 3;
    /*if (options.cartesian_vector) {
      AthenaArray<Real> src;
      src.InitFromTensor(pmb->hydro_u, 4, index::IVX, 3);

      pod = new OutputData;
      pod->type = "VECTORS";
      pod->name = "mom_xyz";
      pod->data.NewAthenaArray(3, pmb->hydro_u.GetDim3(),
                               pmb->hydro_u.GetDim2(), pmb->hydro_u.GetDim1());
      CalculateCartesianVector(src, pod->data, pmb->pcoord);
      AppendOutputDataNode(pod);
      num_vars_ += 3;
    }*/
  }

  // each component of momentum
  if (ContainVariable(options.variable(), "m1")) {
    pod = new OutputData;
    pod->type = "SCALARS";
    pod->name = "mom1";
    pod->data.InitFromTensor(pmb->hydro_u, 4, index::IVX, 1);

    AppendOutputDataNode(pod);
    num_vars_++;
  }
  if (ContainVariable(options.variable(), "m2")) {
    pod = new OutputData;
    pod->type = "SCALARS";
    pod->name = "mom2";
    pod->data.InitFromTensor(pmb->hydro_u, 4, index::IVY, 1);

    AppendOutputDataNode(pod);
    num_vars_++;
  }
  if (ContainVariable(options.variable(), "m3")) {
    pod = new OutputData;
    pod->type = "SCALARS";
    pod->name = "mom3";
    pod->data.InitFromTensor(pmb->hydro_u, 4, index::IVZ, 1);

    AppendOutputDataNode(pod);
    num_vars_++;
  }

  // velocity vector
  if (ContainVariable(options.variable(), "v") ||
      ContainVariable(options.variable(), "prim")) {
    pod = new OutputData;
    pod->type = "VECTORS";
    pod->name = "vel";
    pod->data.InitFromTensor(GET_SHARED("hydro/w"), 4, index::IVX, 3);

    AppendOutputDataNode(pod);
    num_vars_ += 3;
    /*if (options.cartesian_vector) {
      AthenaArray<Real> src;
      src.InitFromTensor(GET_SHARED("hydro/w"), 4, index::IVX, 3);

      pod = new OutputData;
      pod->type = "VECTORS";
      pod->name = "vel_xyz";
      pod->data.NewAthenaArray(3, pmb->phydro_w.GetDim3(),
                               pmb->hydro_w.GetDim2(), pmb->hydro_w.GetDim1());
      CalculateCartesianVector(src, pod->data, pmb->pcoord);
      AppendOutputDataNode(pod);
      num_vars_ += 3;
    }*/
  }

  // each component of velocity
  if (ContainVariable(options.variable(), "vx") ||
      ContainVariable(options.variable(), "v1")) {
    pod = new OutputData;
    pod->type = "SCALARS";
    pod->name = "vel1";
    pod->data.InitFromTensor(GET_SHARED("hydro/w"), 4, index::IVX, 1);

    AppendOutputDataNode(pod);
    num_vars_++;
  }
  if (ContainVariable(options.variable(), "vy") ||
      ContainVariable(options.variable(), "v2")) {
    pod = new OutputData;
    pod->type = "SCALARS";
    pod->name = "vel2";
    pod->data.InitFromTensor(GET_SHARED("hydro/w"), 4, index::IVY, 1);

    AppendOutputDataNode(pod);
    num_vars_++;
  }
  if (ContainVariable(options.variable(), "vz") ||
      ContainVariable(options.variable(), "v3")) {
    pod = new OutputData;
    pod->type = "SCALARS";
    pod->name = "vel3";
    pod->data.InitFromTensor(GET_SHARED("hydro/w"), 4, index::IVZ, 1);

    AppendOutputDataNode(pod);
    num_vars_++;
  }

  // vapor
  auto vapor_cloud =
      peos->pthermo->options.nvapor() + peos->pthermo->options.ncloud();
  if (vapor_cloud > 0) {
    if (options.variable().compare("prim") == 0 ||
        options.variable().compare("vapor") == 0) {
      pod = new OutputData;
      pod->type = "VECTORS";
      pod->name = "vapor";
      pod->data.InitFromTensor(GET_SHARED("hydro/w"), 4, index::ICY,
                               vapor_cloud);

      AppendOutputDataNode(pod);
      num_vars_ += vapor_cloud;
    }

    if (options.variable().compare("cons") == 0) {
      pod = new OutputData;
      pod->type = "VECTORS";
      pod->name = "vapor";
      pod->data.InitFromTensor(pmb->hydro_u, 4, index::ICY, vapor_cloud);

      AppendOutputDataNode(pod);
      num_vars_ += vapor_cloud;
    }
  }
}
}  // namespace snap
