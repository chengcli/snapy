// snap
#include <snap/snap.h>

#include <snap/registry.hpp>

#include "riemann_dispatch.hpp"
#include "riemann_solver.hpp"

namespace snap {

void HLLCSolverImpl::reset() {
  // set up equation-of-state model
  peosl = register_module_op(this, "eosl", options.eos());
  peosr = register_module_op(this, "eosr", options.eos());
}

torch::Tensor HLLCSolverImpl::forward(torch::Tensor wl, torch::Tensor wr,
                                      int dim, torch::Tensor flx) {
  auto el = peosl->compute("W->I", {wl});
  auto gammal = peosl->compute("W->A", {wl});
  auto cl = peosl->compute("WA->L", {wl, gammal});

  auto er = peosr->compute("W->I", {wr});
  auto gammar = peosr->compute("W->A", {wr});
  auto cr = peosr->compute("WA->L", {wr, gammar});

  switch (dim) {
    case 1:
      peosl->pcoord->prim2local3_(wl);
      peosr->pcoord->prim2local3_(wr);
      break;
    case 2:
      peosl->pcoord->prim2local2_(wl);
      peosr->pcoord->prim2local2_(wr);
      break;
    case 3:
      peosl->pcoord->prim2local1_(wl);
      peosr->pcoord->prim2local1_(wr);
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
                  .add_owned_input(el.unsqueeze(0))
                  .add_owned_input(er.unsqueeze(0))
                  .add_owned_const_input(gammal.unsqueeze(0))
                  .add_owned_const_input(gammar.unsqueeze(0))
                  .add_owned_const_input(cl.unsqueeze(0))
                  .add_owned_const_input(cr.unsqueeze(0))
                  .build();

  at::native::call_hllc(flx.device().type(), iter, dim);

  auto pcoord = peosl->pcoord;

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
