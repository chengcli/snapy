// torch
#include <torch/torch.h>

// spdlog
#include <configure.h>
#include <spdlog/sinks/basic_file_sink.h>

// base
#include <globals.h>

// snap
#include <snap/index.h>

#include "eos_formatter.hpp"
#include "equation_of_state.hpp"

namespace snap {
void call_ideal_moist_cpu(at::TensorIterator& iter);
void call_ideal_moist_cuda(at::TensorIterator& iter);

void IdealMoistImpl::reset() {
  // set up thermodynamics model
  pthermo = register_module("thermo", Thermodynamics(options.thermo()));

  // set up coordinate model
  pcoord = register_module_op(this, "coord", options.coord());

  LOG_INFO(logger, "{} resets with options: {}", name(), options);
}

void IdealMoistImpl::cons2prim(torch::Tensor prim, torch::Tensor cons) const {
  if (!HAS_SHARED("hydro/gammad")) {
    SET_SHARED("hydro/gammad") = pthermo->get_gammad(cons, kConserved);
  }

  auto nmass = pthermo->options.nvapor() + pthermo->options.ncloud();
  auto rho = cons[index::IDN] + cons.narrow(0, index::ICY, nmass).sum(0);
  prim[index::IDN] = rho;

  auto feps = pthermo->f_eps(cons / rho);
  auto fsig = pthermo->f_sig(cons / rho);

  auto iter =
      at::TensorIteratorConfig()
          .resize_outputs(false)
          .check_all_same_dtype(true)
          .declare_static_shape(cons.sizes(), /*squash_dims=*/0)
          .add_output(prim)
          .add_input(cons)
          .add_owned_const_input(GET_SHARED("hydro/gammad").unsqueeze(0))
          .add_owned_const_input(feps.unsqueeze(0))
          .add_owned_const_input(fsig.unsqueeze(0))
          .build();

  if (cons.is_cpu()) {
    call_ideal_moist_cpu(iter);
  } else if (cons.is_cuda()) {
    call_ideal_moist_cuda(iter);
  } else {
    cons2prim_fallback(prim, cons, GET_SHARED("hydro/gammad"), feps, fsig);
  }
}

void IdealMoistImpl::cons2prim_fallback(torch::Tensor prim, torch::Tensor cons,
                                        torch::Tensor gammad,
                                        torch::Tensor feps,
                                        torch::Tensor fsig) const {
  _apply_conserved_limiter_inplace(cons);

  auto nmass = pthermo->options.nvapor() + pthermo->options.ncloud();

  // den -> den
  prim[index::IDN] = cons[0] + cons.narrow(0, index::ICY, nmass).sum(0);

  // den -> mixr
  prim.narrow(0, index::ICY, nmass) =
      cons.narrow(0, index::ICY, nmass) / prim[index::IDN];

  // mom -> vel
  prim.narrow(0, index::IVX, 3) =
      cons.narrow(0, index::IVX, 3) / prim[index::IDN];

  pcoord->vec_raise_inplace(prim);

  auto ke =
      0.5 *
      (prim.narrow(0, index::IVX, 3) * cons.narrow(0, index::IVX, 3)).sum(0);

  // eng -> pr
  prim[index::IPR] = (gammad - 1) * (cons[index::IPR] - ke) * feps / fsig;

  _apply_primitive_limiter_inplace(prim);
}

void IdealMoistImpl::prim2cons(torch::Tensor cons, torch::Tensor prim) const {
  _apply_primitive_limiter_inplace(prim);

  auto nmass = pthermo->options.nvapor() + pthermo->options.ncloud();

  // den -> den
  cons[index::IDN] =
      (1. - prim.narrow(0, index::ICY, nmass).sum(0)) * prim[index::IDN];

  // mixr -> den
  cons.narrow(0, index::ICY, nmass) =
      prim.narrow(0, index::ICY, nmass) * prim[index::IDN];

  // vel -> mom
  cons.narrow(0, index::IVX, 3) =
      prim.narrow(0, index::IVX, 3) * prim[index::IDN];

  pcoord->vec_lower_inplace(cons);

  auto ke =
      0.5 *
      (prim.narrow(0, index::IVX, 3) * cons.narrow(0, index::IVX, 3)).sum(0);

  auto gammad = pthermo->get_gammad(prim);

  // pr -> eng
  cons[index::IPR] = prim[index::IPR] * pthermo->f_sig(prim) /
                         pthermo->f_eps(prim) / (gammad - 1) +
                     ke;

  _apply_conserved_limiter_inplace(cons);
}

torch::Tensor IdealMoistImpl::sound_speed(torch::Tensor prim) const {
  auto gammad_m1 = pthermo->get_gammad(prim) - 1;
  return torch::sqrt(
      (1. + gammad_m1 * pthermo->f_eps(prim) / pthermo->f_sig(prim)) *
      prim[index::IPR] / prim[index::IDN]);
}
}  // namespace snap
