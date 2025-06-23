// kintere
#include <kintera/thermo/eval_uhs.hpp>
#include <kintera/thermo/thermo.hpp>

// snap
#include <snap/snap.h>

#include <snap/registry.hpp>

#include "equation_of_state.hpp"

namespace snap {

MoistMixtureImpl::MoistMixtureImpl(EquationOfStateOptions const &options_)
    : EquationOfStateImpl(options_) {
  reset();
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
      "W", torch::empty({nhydro(), nx3, nx2, nx1}, torch::kFloat64));

  _cons = register_buffer(
      "U", torch::empty({nhydro(), nx3, nx2, nx1}, torch::kFloat64));

  _gamma = register_buffer("A", torch::empty({nx3, nx2, nx1}, torch::kFloat64));

  _ct = register_buffer("N", torch::empty({nx3, nx2, nx1}, torch::kFloat64));

  _cs = register_buffer("L", torch::empty({nx3, nx2, nx1}, torch::kFloat64));
}

torch::Tensor MoistMixtureImpl::compute(
    std::string ab, std::vector<torch::Tensor> const &args) {
  if (ab == "W->U") {
    _prim.set_(args[0]);
    _prim2cons(_prim, _cons);
    return _cons;
  } else if (ab == "U->W") {
    _cons.set_(args[0]);
    _cons2prim(_cons, _prim);
    return _prim;
  } else if (ab == "W->A") {
    auto dens = args[0][Index::IDN];
    auto pres = args[0][Index::IPR];
    int ny = pthermo->options.vapor_ids().size() +
             pthermo->options.cloud_ids().size() - 1;
    auto yfrac = args[0].narrow(0, Index::ICY, ny);
    auto ivol = pthermo->compute("DY->V", {dens, yfrac});
    auto temp = pthermo->compute("PV->T", {pres, ivol});
    _adiabatic_index(temp, ivol, _gamma);
    return _gamma;
  } else if (ab == "TV->A") {
    _adiabatic_index(args[0], args[1], _gamma);
    return _gamma;
  } else if (ab == "TV->N") {
    _isothermal_sound_speed(args[0], args[1], _ct);
    return _ct;
  } else if (ab == "W->L") {
    _adiabatic_index(args[0], args[1], _gamma);
    _isothermal_sound_speed(args[0], args[1], _ct);
    torch::mul_out(_cs, _gamma.sqrt(), _ct);
    return _cs;
  } else {
    TORCH_CHECK(false, "Unknown abbreviation: ", ab);
  }
}

void MoistMixtureImpl::_prim2cons(torch::Tensor prim, torch::Tensor &cons) {
  _apply_primitive_limiter_(prim);
  int ny = pthermo->options.vapor_ids().size() +
           pthermo->options.cloud_ids().size() - 1;

  // den -> den
  auto out = cons[Index::IDN];
  torch::mul_out(/*out=*/out, (1. - prim.narrow(0, Index::ICY, ny).sum(0)),
                 prim[Index::IDN]);

  // mixr -> den
  out = cons.narrow(0, Index::ICY, ny);
  torch::mul_out(/*out=*/out, prim.narrow(0, Index::ICY, ny), prim[Index::IDN]);

  // vel -> mom
  out = cons.narrow(0, Index::IVX, 3);
  torch::mul_out(/*out=*/out, prim.narrow(0, Index::IVX, 3), prim[Index::IDN]);

  pcoord->vec_lower_(cons);

  // KE
  auto tmp = prim.narrow(0, Index::IVX, 3) * cons.narrow(0, Index::IVX, 3);
  out = cons[Index::IPR];
  torch::sum_out(/*out=*/out, tmp, /*dim=*/0);
  out *= 0.5;

  auto ivol = pthermo->compute(
      "DY->V", {prim[Index::IDN], prim.narrow(0, Index::ICY, ny)});
  auto temp = pthermo->compute("PV->T", {prim[Index::IPR], ivol});

  // KE + intEng
  cons[Index::IPR] += pthermo->compute("VT->U", {ivol, temp});

  _apply_conserved_limiter_(cons);
}

void MoistMixtureImpl::_cons2prim(torch::Tensor cons, torch::Tensor &prim) {
  _apply_conserved_limiter_(cons);
  int ny = pthermo->options.vapor_ids().size() +
           pthermo->options.cloud_ids().size() - 1;

  // den -> den
  auto out = prim[Index::IDN];
  torch::sum_out(/*out=*/out, cons.narrow(0, Index::ICY, ny), /*dim=*/0);
  out += cons[Index::IDN];

  // den -> mixr
  out = prim.narrow(0, Index::ICY, ny);
  torch::div_out(/*out=*/out, cons.narrow(0, Index::ICY, ny), prim[Index::IDN]);

  // mom -> vel
  out = prim.narrow(0, Index::IVX, 3);
  torch::div_out(/*out=*/out, cons.narrow(0, Index::IVX, 3), prim[Index::IDN]);

  pcoord->vec_raise_(prim);

  // KE (TODO: cli, new kernel for this operation)
  auto KE =
      (prim.narrow(0, Index::IVX, 3) * cons.narrow(0, Index::IVX, 3)).sum(0);
  KE *= 0.5;

  auto ivol = pthermo->compute(
      "DY->V", {prim[Index::IDN], prim.narrow(0, Index::ICY, ny)});
  auto temp = pthermo->compute("VU->T", {ivol, cons[Index::IPR] - KE});
  prim[Index::IPR] = pthermo->compute("VT->P", {ivol, temp});

  _apply_primitive_limiter_(prim);
}

void MoistMixtureImpl::_adiabatic_index(torch::Tensor temp, torch::Tensor ivol,
                                        torch::Tensor &out) const {
  auto conc = ivol * pthermo->inv_mu;
  auto cp = kintera::eval_cp_R(temp, conc, pthermo->options);
  auto cv = kintera::eval_cv_R(temp, conc, pthermo->options);

  auto cp_vol = (conc * cp).sum(-1);
  auto cv_vol = (conc * cv).sum(-1);
  torch::div_out(/*out=*/out, cp_vol, cv_vol);
}

void MoistMixtureImpl::_isothermal_sound_speed(torch::Tensor prim,
                                               torch::Tensor ivol,
                                               torch::Tensor &out) const {}

}  // namespace snap
