// torch
#include <torch/torch.h>

// base
#include <configure.h>

// snap
#include <snap/snap.h>

#include "eos_formatter.hpp"
#include "equation_of_state.hpp"

namespace snap {
void call_ideal_moist_cpu(at::TensorIterator& iter);
__attribute__((weak)) void call_ideal_moist_cuda(at::TensorIterator& iter) {}

void IdealMoistImpl::reset() {
  // set up thermodynamics model
  pthermo = register_module("thermo", kintera::ThermoY(options.thermo()));

  // set up coordinate model
  pcoord = register_module_op(this, "coord", options.coord());
}

torch::Tensor IdealMoistImpl::cons2prim(
    torch::Tensor cons, torch::optional<torch::Tensor> out) const;
if (!HAS_SHARED("eos/gammad")) {
  SET_SHARED("eos/gammad") =
      pthermo->options.gammad() * torch::ones_like(cons[Index::IDN]);
}
auto prim = out.value_or(torch::empty_like(cons));

auto ny =
    pthermo->options.vapor_ids().size() + pthermo->options.cloud_ids().size();
auto rho = cons[Index::IDN] + cons.narrow(0, Index::ICY, ny).sum(0);
prim[Index::IDN] = rho;

auto feps = pthermo->f_eps(cons / rho, /*start=*/Index::ICY);
auto fsig = pthermo->f_sig(cons / rho, /*start=*/Index::ICY);

auto iter = at::TensorIteratorConfig()
                .resize_outputs(false)
                .check_all_same_dtype(true)
                .declare_static_shape(cons.sizes(), /*squash_dims=*/0)
                .add_output(prim)
                .add_input(cons)
                .add_owned_const_input(GET_SHARED("eos/gammad").unsqueeze(0))
                .add_owned_const_input(feps.unsqueeze(0))
                .add_owned_const_input(fsig.unsqueeze(0))
                .build();

if (cons.is_cpu()) {
  call_ideal_moist_cpu(iter);
} else if (cons.is_cuda()) {
  call_ideal_moist_cuda(iter);
} else {
  _cons2prim_fallback(prim, cons, GET_SHARED("eos/gammad"), feps, fsig);
}

return prim;
}

torch::Tensor IdealMoistImpl::prim2cons(
    torch::Tensor prim, torch::optional<torch::Tensor> out) const {
  //_apply_primitive_limiter_inplace(prim);
  auto cons = out.value_or(torch::empty_like(prim));

  auto ny =
      pthermo->options.vapor_ids().size() + pthermo->options.cloud_ids().size();

  // den -> den
  cons[Index::IDN] =
      (1. - prim.narrow(0, Index::ICY, ny).sum(0)) * prim[Index::IDN];

  // mixr -> den
  cons.narrow(0, Index::ICY, ny) =
      prim.narrow(0, Index::ICY, ny) * prim[Index::IDN];

  // vel -> mom
  cons.narrow(0, Index::IVX, 3) =
      prim.narrow(0, Index::IVX, 3) * prim[Index::IDN];

  pcoord->vec_lower_inplace(cons);

  auto ke =
      0.5 *
      (prim.narrow(0, Index::IVX, 3) * cons.narrow(0, Index::IVX, 3)).sum(0);

  auto gammad = pthermo->options.gammad();
  auto feps = pthermo->f_eps(prim, /*start=*/Index::ICY);
  auto fsig = pthermo->f_sig(prim, /*start*/ Index::ICY);

  // pr -> eng
  cons[Index::IPR] = prim[Index::IPR] * fsig / feps / (gammad - 1) + ke;

  _apply_conserved_limiter_inplace(cons);

  return cons;
}

torch::Tensor IdealMoistImpl::sound_speed(
    torch::Tensor prim, torch::optional<torch::Tensor> out) const {
  auto gammad_m1 = pthermo->options.gammad() - 1;
  auto cs = out.value_or(torch::empty_like(prim[Index::IDN]));
  auto feps = pthermo->f_eps(prim, /*start=*/Index::ICY);
  auto fsig = pthermo->f_sig(prim, /*start=*/Index::ICY);

  cs = torch::sqrt((1. + gammad_m1 * feps / fsig) * prim[Index::IPR] /
                   prim[Index::IDN]);
  return cs;
}

void IdealMoistImpl::_cons2prim_fallback(torch::Tensor prim, torch::Tensor cons,
                                         torch::Tensor gammad,
                                         torch::Tensor feps,
                                         torch::Tensor fsig) const {
  _apply_conserved_limiter_inplace(cons);

  auto ny =
      pthermo->options.vapor_ids().size() + pthermo->options.cloud_ids().size();

  // den -> den
  prim[Index::IDN] = cons[0] + cons.narrow(0, Index::ICY, ny).sum(0);

  // den -> mixr
  prim.narrow(0, Index::ICY, ny) =
      cons.narrow(0, Index::ICY, ny) / prim[Index::IDN];

  // mom -> vel
  prim.narrow(0, Index::IVX, 3) =
      cons.narrow(0, Index::IVX, 3) / prim[Index::IDN];

  pcoord->vec_raise_inplace(prim);

  auto ke =
      0.5 *
      (prim.narrow(0, Index::IVX, 3) * cons.narrow(0, Index::IVX, 3)).sum(0);

  // eng -> pr
  prim[Index::IPR] = (gammad - 1) * (cons[Index::IPR] - ke) * feps / fsig;

  _apply_primitive_limiter_inplace(prim);
}

}  // namespace snap
