// kintera
#include <kintera/thermo/eval_uhs.hpp>
#include <kintera/thermo/thermo.hpp>

// snap
#include <snap/snap.h>

#include <snap/registry.hpp>

#include "equation_of_state.hpp"

namespace snap {

MoistMixtureImpl::MoistMixtureImpl(EquationOfStateOptions const &options_)
    : EquationOfStateImpl(options_) {
  reset()
}

void MoistMixtureImpl::reset() {
  // set up coordinate model
  pcoord = register_module_op(this, "coord", options.coord());

  // set up thermodynamics model
  pthermo = register_module("thermo", kintera::ThermoY(options.thermo()));

  // populate buffers
  int nx1 = options.coord().nx1();
  int nx2 = options.coord().nx2();
  int nx3 = options.coord().nx3();

  _prim = register_buffer(
      "prim", torch::empty({nhydro(), nx3, nx2, nx1}, torch::kFloat64));

  _cons = register_buffer(
      "cons", torch::empty({nhydro(), nx3, nx2, nx1}, torch::kFloat64));

  _gamma =
      register_buffer("gamma", torch::empty({nx3, nx2, nx1}, torch::kFloat64));

  _ct = register_buffer("ct", torch::empty({nx3, nx2, nx1}, torch::kFloat64));
}

torch::Tensor const &MoistMixture::compute(
    std::string ab, std::vector<torch::Tensor> const &args) {
  if (ab == "prim->cons") {
    _prim.set_(args[0]);
    _prim2cons(_prim, _cons);
    return _cons;
  } else if (ab == "cons->prim") {
    _cons.set_(args[0]);
    _cons2prim(_cons, _prim);
    return _prim;
  } else if (ab == "TV->gamma") {
    _adiabatic_index(args[0], args[1], _gamma);
    return _gamma;
  } else if (ab == "TV->ct") {
    _isothermal_sound_speed(args[0], args[1], _ct);
    return _ct;
  } else if (ab == "TV->cs") {
    _adiabatic_index(args[0], args[1], _gamma);
    _isothermal_sound_speed(args[0], args[1], _ct);
    return _gamma.sqrt() * _ct;
  } else {
    TORCH_CHECK(false, "Unknown abbreviation: ", ab);
  }
}

void MoistMixtureImpl::prim2cons(torch::Tensor prim,
                                 torch::Tensor &cons) const {
  _apply_primitive_limiter_(prim);
  int ny = pthermo->options.vapor_ids().size() +
           pthermo->options.cloud_ids().size() - 1;

  // den -> den
  cons[Index::IDN] =
      (1. - prim.narrow(0, Index::ICY, ny).sum(0)) * prim[Index::IDN];

  // mixr -> den
  cons.narrow(0, Index::ICY, ny) =
      prim.narrow(0, Index::ICY, ny) * prim[Index::IDN];

  // vel -> mom
  cons.narrow(0, Index::IVX, 3) =
      prim.narrow(0, Index::IVX, 3) * prim[Index::IDN];

  pcoord->vec_lower_(cons);

  // KE
  cons[Index::IPR] =
      0.5 *
      (prim.narrow(0, Index::IVX, 3) * cons.narrow(0, Index::IVX, 3)).sum(0);

  auto ivol = pthermo->compute("DY->V", prim[Index::IDN],
                               prim.narrow(0, Index::ICY, ny));
  auto temp = pthermo->compute("PV->T", prim[Index::IPR], ivol);

  // KE + intEng
  cons[Index::IPX] += pthermo->compute("VT->U", ivol, temp);

  _apply_conserved_limiter_(cons);
}

void MoistMixtureImpl::_cons2prim(torch::Tensor cons,
                                  torch::Tensor &prim) const {
  _apply_conserved_limiter_(cons);
  int ny = pthermo->options.vapor_ids().size() +
           pthermo->options.cloud_ids().size() - 1;

  // den -> den
  prim[Index::IDN] = cons[0] + cons.narrow(0, Index::ICY, ny).sum(0);

  // den -> mixr
  prim.narrow(0, Index::ICY, ny) =
      cons.narrow(0, Index::ICY, ny) / prim[Index::IDN];

  // mom -> vel
  prim.narrow(0, Index::IVX, 3) =
      cons.narrow(0, Index::IVX, 3) / prim[Index::IDN];

  pcoord->vec_raise_(prim);

  auto KE =
      0.5 *
      (prim.narrow(0, Index::IVX, 3) * cons.narrow(0, Index::IVX, 3)).sum(0);

  auto ivol = pthermo->compute("DY->V", prim[Index::IDN],
                               prim.narrow(0, Index::ICY, ny));
  auto temp = pthermo->compute("VU->T", ivol, cons[Index::IPX] - KE);
  prim[Index::IPR] = pthermo->compute("VT->P", ivol, temp);

  _apply_primitive_limiter_(prim);
}

void MoistMixture::_adiabatic_index(torch::Tensor temp, torch::Tensor ivol,
                                    torch::Tensor &out) const {
  auto conc = ivol * pthermo->inv_mu;
  auto cp = eval_cp_R(temp, conc, options) auto cv =
      eval_cv_R(temp, conc, options)

          auto cp_vol = (conc * cp).sum(-1);
  auto cv_vol = (conc * cv).sum(-1);
  out.set_(cp_vol / cv_vol);
}

void MoistMixture::_isothermal_sound_speed(torch::Tensor prim,
                                           torch::Tensor ivol,
                                           torch::Tensor &out) const {}
