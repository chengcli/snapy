// snap
#include <snap/snap.h>

#include <snap/coord/cubed_sphere_utils.hpp>
#include <snap/hydro/hydro.hpp>
#include <snap/layout/cubed_sphere_layout.hpp>
#include <snap/mesh/meshblock.hpp>

#include "output_type.hpp"
#include "output_utils.hpp"

namespace snap {

void OutputType::loadHydroOutputData(MeshBlockImpl* pmb,
                                     Variables const& vars) {
  OutputData* pod;

  auto peos = pmb->phydro->peos;
  auto pcoord = pmb->pcoord;

  auto const& w = vars.at("hydro_w");
  auto const& u = vars.at("hydro_u");

  // (lab-frame) density
  if (ContainVariable("D") || ContainVariable("cons")) {
    pod = new OutputData;
    pod->type = "SCALARS";
    pod->name = "dens";
    pod->data.InitFromTensor(u, 4, IDN, 1);
    AppendOutputDataNode(pod);
    num_vars_++;
  }

  // (rest-frame) density
  if (ContainVariable("d") || ContainVariable("prim")) {
    pod = new OutputData;
    pod->type = "SCALARS";
    pod->name = "rho";
    pod->data.InitFromTensor(w, 4, IDN, 1);
    AppendOutputDataNode(pod);
    num_vars_++;
  }

  // total energy
  if (peos->nvar() > 4) {
    if (ContainVariable("E") || ContainVariable("cons")) {
      pod = new OutputData;
      pod->type = "SCALARS";
      pod->name = "Etot";
      pod->data.InitFromTensor(u, 4, IPR, 1);

      AppendOutputDataNode(pod);
      num_vars_++;
    }

    // pressure
    if (ContainVariable("p") || ContainVariable("prim")) {
      pod = new OutputData;
      pod->type = "SCALARS";
      pod->name = "press";
      pod->data.InitFromTensor(w, 4, IPR, 1);
      AppendOutputDataNode(pod);
      num_vars_++;
    }
  }

  // momentum vector
  if (ContainVariable("m") || ContainVariable("cons")) {
    pod = new OutputData;
    pod->type = "VECTORS";
    pod->name = "mom";
    pod->data.InitFromTensor(u, 4, IVX, 3);

    AppendOutputDataNode(pod);
    num_vars_ += 3;
    /*if (options.cartesian_vector) {
      AthenaArray<Real> src;
      src.InitFromTensor(pmb->hydro_u, 4, IVX, 3);

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
  if (ContainVariable("m1")) {
    pod = new OutputData;
    pod->type = "SCALARS";
    pod->name = "mom1";
    pod->data.InitFromTensor(u, 4, IVX, 1);

    AppendOutputDataNode(pod);
    num_vars_++;
  }
  if (ContainVariable("m2")) {
    pod = new OutputData;
    pod->type = "SCALARS";
    pod->name = "mom2";
    pod->data.InitFromTensor(u, 4, IVY, 1);

    AppendOutputDataNode(pod);
    num_vars_++;
  }
  if (ContainVariable("m3")) {
    pod = new OutputData;
    pod->type = "SCALARS";
    pod->name = "mom3";
    pod->data.InitFromTensor(u, 4, IVZ, 1);

    AppendOutputDataNode(pod);
    num_vars_++;
  }

  // velocity vector
  if (ContainVariable("v") || ContainVariable("prim")) {
    pod = new OutputData;
    pod->type = "VECTORS";
    pod->name = "vel";
    pod->data.InitFromTensor(w, 4, IVX, 3);

    AppendOutputDataNode(pod);
    num_vars_ += 3;
    /*if (options.cartesian_vector) {
      AthenaArray<Real> src;
      src.InitFromTensor(GET_SHARED("hydro/w"), 4, IVX, 3);

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
  if (ContainVariable("vx") || ContainVariable("v1")) {
    pod = new OutputData;
    pod->type = "SCALARS";
    pod->name = "vel1";
    pod->data.InitFromTensor(w, 4, IVX, 1);

    AppendOutputDataNode(pod);
    num_vars_++;
  }
  if (ContainVariable("vy") || ContainVariable("v2")) {
    pod = new OutputData;
    pod->type = "SCALARS";
    pod->name = "vel2";
    pod->data.InitFromTensor(w, 4, IVY, 1);

    AppendOutputDataNode(pod);
    num_vars_++;
  }
  if (ContainVariable("vz") || ContainVariable("v3")) {
    pod = new OutputData;
    pod->type = "SCALARS";
    pod->name = "vel3";
    pod->data.InitFromTensor(w, 4, IVZ, 1);

    AppendOutputDataNode(pod);
    num_vars_++;
  }

  // vapor + cloud
  auto ny = peos->nvar() - 5;
  if (ny > 0) {
    if (ContainVariable("prim")) {
      pod = new OutputData;
      pod->type = "VECTORS";
      pod->name = get_hydro_names(pmb);
      pod->data.InitFromTensor(w, 4, ICY, ny);

      AppendOutputDataNode(pod);
      num_vars_ += ny;
    }

    if (ContainVariable("cons")) {
      pod = new OutputData;
      pod->type = "VECTORS";
      pod->name = get_hydro_names(pmb);
      pod->data.InitFromTensor(u, 4, ICY, ny);

      AppendOutputDataNode(pod);
      num_vars_ += ny;
    }
  }

  // lat/lon grid for cubed sphere
  if (ContainVariable("prim") || ContainVariable("cons")) {
    if (pcoord->options->type() == "gnomonic-equiangle") {
      int r = pmb->options->layout()->rank();
      auto [rx, ry, face_id] = pmb->get_layout()->loc_of(r);
      auto face = CS_FACE_NAMES[face_id];

      auto mesh = torch::meshgrid({pcoord->x3v, pcoord->x2v}, "ij");
      auto alpha = mesh[1];
      auto beta = mesh[0];
      auto [lon, lat] = cs_ab_to_lonlat(face, alpha, beta);

      // longitude
      pod = new OutputData;
      pod->type = "SCALARS";
      pod->name = "lon";
      pod->data.CopyFromTensor(lon);
      AppendOutputDataNode(pod);
      num_vars_++;

      // latitude
      pod = new OutputData;
      pod->type = "SCALARS";
      pod->name = "lat";
      pod->data.CopyFromTensor(lat);
      AppendOutputDataNode(pod);
      num_vars_++;
    }
  }
}
}  // namespace snap
