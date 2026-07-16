// base
#include <configure.h>

#include <snap/coord/coordinate.hpp>
#include <snap/hydro/hydro.hpp>
#include <snap/mesh/meshblock.hpp>

// snap
#include "riemann_dispatch.hpp"
#include "riemann_solver.hpp"

namespace snap {

void ShallowRoeSolverImpl::reset() {
  TORCH_CHECK(phydro, "[ShallowRoeSolver] Parent Hydro is null");
  auto pcoord = phydro->pmb->pcoord;

  if (pcoord->options->type() == "gnomonic_equiangle") {
    TORCH_CHECK(options->dir() == "yz",
                "ShallowRoeSolver with GnomonicEquiangle coordinate "
                "only supports options->dir() = 'yz' but got options->dir() = ",
                options->dir());
  }
}

torch::Tensor ShallowRoeSolverImpl::forward(torch::Tensor wl, torch::Tensor wr,
                                            int dim, torch::Tensor flx,
                                            torch::Tensor face_pressure) {
  TORCH_CHECK(!face_pressure.defined(),
              "Face-pressure output is not implemented for ShallowRoeSolver");
  auto pcoord = phydro->pmb->pcoord;

  if (options->dir() != "xy" && options->dir() != "yz") {
    TORCH_CHECK(false,
                "ShallowRoeSolver takes options->dir() = 'xy' or 'yz'"
                " but got options->dir() = ",
                options->dir());
  }

  switch (dim) {
    case 1:
      pcoord->prim2local3_(wl);
      pcoord->prim2local3_(wr);
      break;
    case 2:
      pcoord->prim2local2_(wl);
      pcoord->prim2local2_(wr);
      break;
    case 3:
      pcoord->prim2local1_(wl);
      pcoord->prim2local1_(wr);
      break;
    default:
      TORCH_CHECK(false, "Invalid dimension: ", dim);
  }

  auto iter = at::TensorIteratorConfig()
                  .resize_outputs(false)
                  .check_all_same_dtype(true)
                  .declare_static_shape(flx.sizes(), /*squash_dims=*/0)
                  .add_output(flx)
                  .add_input(wl)
                  .add_input(wr)
                  .build();

  at::native::call_shallow_roe(flx.device().type(), iter, dim,
                               options->dir() == "yz");

  switch (dim) {
    case 1:
      pcoord->flux2global3_(flx);
      break;
    case 2:
      pcoord->flux2global2_(flx);
      break;
    case 3:
      pcoord->flux2global1_(flx);
      break;
    default:
      TORCH_CHECK(false, "Invalid dimension: ", dim);
  }

  return flx;
}
}  // namespace snap
