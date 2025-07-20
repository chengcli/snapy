// snap
#include <snap/snap.h>

#include <snap/registry.hpp>

#include "riemann_dispatch.hpp"
#include "riemann_solver.hpp"

namespace snap {

void LmarsSolverImpl::reset() {
  // set up equation-of-state model
  peos = register_module_op(this, "eos", options.eos());

  // register buffers
  auto vec = peos->get_buffer("W").sizes().vec();
  vec[0] = 2;

  elr = register_buffer("elr", torch::empty(vec, torch::kFloat64));
  clr = register_buffer("clr", torch::empty(vec, torch::kFloat64));
  glr = register_buffer("glr", torch::empty(vec, torch::kFloat64));
}

torch::Tensor LmarsSolverImpl::forward(torch::Tensor wl, torch::Tensor wr,
                                       int dim, torch::Tensor flx) {
  elr[ILT] = peos->compute("W->I", {wl}) / wl[Index::IDN];
  glr[ILT] = peos->compute("W->A", {wl});

  elr[IRT] = peos->compute("W->I", {wr}) / wr[Index::IDN];
  glr[IRT] = peos->compute("W->A", {wr});

  auto pcoord = peos->pcoord;

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
                  .add_input(elr)
                  .add_input(glr)
                  .build();

  at::native::call_lmars(flx.device().type(), iter, dim);

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
