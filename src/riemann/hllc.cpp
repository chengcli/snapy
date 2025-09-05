// snap
#include <snap/snap.h>

#include <snap/registry.hpp>

#include "riemann_dispatch.hpp"
#include "riemann_solver.hpp"

namespace snap {

void HLLCSolverImpl::reset() {
  // set up equation-of-state model
  peos = register_module_op(this, "eos", options.eos());

  // register buffers
  auto nc1 = peos->pcoord->options.nc1();
  auto nc2 = peos->pcoord->options.nc2();
  auto nc3 = peos->pcoord->options.nc3();

  elr =
      register_buffer("elr", torch::empty({2, nc3, nc2, nc1}, torch::kFloat64));
  clr =
      register_buffer("clr", torch::empty({2, nc3, nc2, nc1}, torch::kFloat64));
  glr =
      register_buffer("glr", torch::empty({2, nc3, nc2, nc1}, torch::kFloat64));
}

torch::Tensor HLLCSolverImpl::forward(torch::Tensor wl, torch::Tensor wr,
                                      int dim, torch::Tensor flx) {
  auto pcoord = peos->pcoord;

  elr[ILT] = peos->compute("W->I", {wl});

  if (options.eos().type() == "aneos") {
    clr[ILT] = peos->compute("W->L", {wl});
    glr[ILT] = peos->compute("WL->A", {wl, clr[ILT]});
  } else {
    glr[ILT] = peos->compute("W->A", {wl});
    clr[ILT] = peos->compute("WA->L", {wl, glr[ILT]});
  }

  elr[IRT] = peos->compute("W->I", {wr});

  if (options.eos().type() == "aneos") {
    clr[IRT] = peos->compute("W->L", {wr});
    glr[IRT] = peos->compute("WL->A", {wr, clr[IRT]});
  } else {
    glr[IRT] = peos->compute("W->A", {wr});
    clr[IRT] = peos->compute("WA->L", {wr, glr[IRT]});
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
                  .check_all_same_dtype(false)
                  .declare_static_shape(flx.sizes(), /*squash_dims=*/0)
                  .add_output(flx)
                  .add_input(wl)
                  .add_input(wr)
                  .add_input(elr)
                  .add_input(glr)
                  .add_input(clr)
                  .build();

  at::native::call_hllc(flx.device().type(), iter, dim);

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
