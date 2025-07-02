// snap
#include <snap/snap.h>

#include <snap/registry.hpp>

#include "riemann_dispatch.hpp"
#include "riemann_solver.hpp"

namespace snap {

void LmarsSolverImpl::reset() {
  // set up equation-of-state model
  peosl = register_module_op(this, "eosl", options.eos());
  peosr = register_module_op(this, "eosr", options.eos());
}

torch::Tensor LmarsSolverImpl::forward(torch::Tensor wl, torch::Tensor wr,
                                       int dim, torch::Tensor flx) {
  wl[IDN].clamp_min_(options.eos().density_floor());
  wl[IPR].clamp_min_(options.eos().pressure_floor());

  wr[IDN].clamp_min_(options.eos().density_floor());
  wr[IPR].clamp_min_(options.eos().pressure_floor());

  auto el = peosl->compute("W->I", {wl}) / wl[Index::IDN];
  auto gammal = peosl->compute("W->A", {wl});

  auto er = peosr->compute("W->I", {wr}) / wr[Index::IDN];
  auto gammar = peosr->compute("W->A", {wr});

  peosl->pcoord->prim2local_(wl);
  peosr->pcoord->prim2local_(wr);

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

  peosl->pcoord->flux2global_(flx);

  return flx;
}

}  // namespace snap
