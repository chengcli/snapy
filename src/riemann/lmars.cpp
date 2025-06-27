// snap
#include <snap/snap.h>

#include <snap/registry.hpp>

#include "riemann_dispatch.hpp"
#include "riemann_solver.hpp"

namespace snap {

void LmarsSolverImpl::reset() {
  // set up equation-of-state model
  peos = register_module_op(this, "eos", options.eos());
}

torch::Tensor LmarsSolverImpl::forward(torch::Tensor wl, torch::Tensor wr,
                                       int dim, torch::Tensor dummy) {
  auto flx = torch::empty_like(wl);

  auto el = peos->compute("W->U", {wl}) / wl[Index::IDN];
  auto gammal = peos->compute("W->A", {wl});

  auto er = peos->compute("W->U", {wr}) / wr[Index::IDN];
  auto gammar = peos->compute("W->A", {wr});

  peos->pcoord->prim2local_(wl);

  auto iter = at::TensorIteratorConfig()
                  .resize_outputs(false)
                  .check_all_same_dtype(true)
                  .declare_static_shape(flx.sizes(), /*squash_dims=*/0)
                  .add_output(flx)
                  .add_input(wl)
                  .add_input(wr)
                  .add_owned_input(el.unsqueeze(0))
                  .add_owned_input(er.unsqueeze(0))
                  .add_owned_const_input(gammal.unsqueeze(0))
                  .add_owned_const_input(gammar.unsqueeze(0))
                  .build();

  at::native::call_lmars(flx.device().type(), iter, dim);

  peos->pcoord->flux2global_(flx);

  return flx;
}

}  // namespace snap
