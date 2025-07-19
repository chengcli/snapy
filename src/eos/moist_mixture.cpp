// kintere
#include <kintera/constants.h>

#include <kintera/thermo/eval_uhs.hpp>
#include <kintera/thermo/thermo.hpp>

// snap
#include <snap/snap.h>

#include <snap/registry.hpp>

#include "moist_mixture.hpp"

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
  int nc1 = options.coord().nc1();
  int nc2 = options.coord().nc2();
  int nc3 = options.coord().nc3();

  _prim = register_buffer(
      "W", torch::empty({nvar(), nc3, nc2, nc1}, torch::kFloat64));

  _cons = register_buffer(
      "U", torch::empty({nvar(), nc3, nc2, nc1}, torch::kFloat64));

  _gamma = register_buffer("A", torch::empty({nc3, nc2, nc1}, torch::kFloat64));

  _ct = register_buffer("N", torch::empty({nc3, nc2, nc1}, torch::kFloat64));

  _cs = register_buffer("L", torch::empty({nc3, nc2, nc1}, torch::kFloat64));

  _ke = register_buffer("K", torch::empty({nc3, nc2, nc1}, torch::kFloat64));

  _ie = register_buffer("I", torch::empty({nc3, nc2, nc1}, torch::kFloat64));

  int ny = std::max(0, (int)nvar() - 5);
  _ce =
      register_buffer("E", torch::empty({ny, nc3, nc2, nc1}, torch::kFloat64));
  _rhoc =
      register_buffer("C", torch::empty({ny, nc3, nc2, nc1}, torch::kFloat64));
}

torch::Tensor MoistMixtureImpl::compute(
    std::string ab, std::vector<torch::Tensor> const &args) {
  if (ab == "W->U") {
    _prim2cons(args[0], _cons);
    return _cons;
  } else if (ab == "W->I") {
    _prim2intEng(args[0], _ie);
    return _ie;
  } else if (ab == "W->T") {
    auto temp = get_buffer("thermo.T");
    _prim2temp(args[0], temp);
    return temp;
  } else if (ab == "W->E") {
    _prim2cloudEng(args[0], _ce);
    return _ce;
  } else if (ab == "U->W") {
    _cons2prim(args[0], _prim);
    return _prim;
  } else if (ab == "U->K") {
    _cons2ke(args[0], _ke);
    return _ke;
  } else if (ab == "W->A") {
    auto ivol = pthermo->named_buffers()["V"];
    auto temp = pthermo->named_buffers()["T"];
    _adiabatic_index(ivol, temp, _gamma);
    return _gamma;
  } else if (ab == "WA->L") {
    auto ivol = pthermo->named_buffers()["V"];
    auto temp = pthermo->named_buffers()["T"];
    _isothermal_sound_speed(ivol, temp, args[0][IDN], _ct);
    torch::mul_out(_cs, args[1].sqrt(), _ct);
    return _cs;
  } else {
    TORCH_CHECK(false, "Unknown abbreviation: ", ab);
  }
}

void MoistMixtureImpl::_prim2intEng(torch::Tensor prim, torch::Tensor &ie) {
  int ny = pthermo->options.vapor_ids().size() +
           pthermo->options.cloud_ids().size() - 1;

  auto ivol = pthermo->compute("DY->V", {prim[IDN], prim.narrow(0, ICY, ny)});
  auto temp = pthermo->compute("PV->T", {prim[IPR], ivol});
  ie.set_(pthermo->compute("VT->U", {ivol, temp}));
}

void MoistMixtureImpl::_prim2cons(torch::Tensor prim, torch::Tensor &cons) {
  apply_primitive_limiter_(prim);
  int ny = pthermo->options.vapor_ids().size() +
           pthermo->options.cloud_ids().size() - 1;

  // den -> den
  auto out = cons[IDN];
  torch::mul_out(out, (1. - prim.narrow(0, ICY, ny).sum(0)), prim[IDN]);

  // mixr -> den
  out = cons.narrow(0, ICY, ny);
  torch::mul_out(out, prim.narrow(0, ICY, ny), prim[IDN]);

  // vel -> mom
  out = cons.narrow(0, IVX, 3);
  torch::mul_out(out, prim.narrow(0, IVX, 3), prim[IDN]);

  pcoord->vec_lower_(out);

  // KE
  _ke.set_((prim.narrow(0, IVX, 3) * cons.narrow(0, IVX, 3)).sum(0));
  _ke *= 0.5;

  // IE
  _prim2intEng(prim, _ie);

  out = cons[IPR];
  torch::add_out(out, _ke, _ie);

  apply_conserved_limiter_(cons);
}

void MoistMixtureImpl::_prim2temp(torch::Tensor prim, torch::Tensor &out) {
  int ny = pthermo->options.vapor_ids().size() +
           pthermo->options.cloud_ids().size() - 1;
  auto yfrac = prim.narrow(0, ICY, ny);
  auto ivol = pthermo->compute("DY->V", {prim[IDN], yfrac});
  out.set_(pthermo->compute("PV->T", {prim[IPR], ivol}));
}

void MoistMixtureImpl::_prim2cloudEng(torch::Tensor prim, torch::Tensor &out) {
  int nvapor = pthermo->options.vapor_ids().size() - 1;
  int ncloud = pthermo->options.cloud_ids().size();
  int ny = nvapor + ncloud;

  auto yfrac = prim.narrow(0, ICY, ny);
  auto ivol = pthermo->compute("DY->V", {prim[IDN], yfrac});
  auto temp = pthermo->compute("PV->T", {prim[IPR], ivol});

  auto Rgas = kintera::constants::Rgas * pthermo->inv_mu;
  auto ie = eval_intEng_R(temp, ivol, pthermo->options) * Rgas * ivol;

  auto vel = prim.narrow(0, IVX, 3).clone();

  pcoord->vec_lower_(vel);
  auto ke = 0.5 * (prim.narrow(0, IVX, 3) * vel).sum(0, /*keepdim=*/true);

  _rhoc.set_(prim[IDN] * yfrac);
  out.set_(ie.narrow(-1, 1, ny).permute({3, 0, 1, 2}) + ke * _rhoc);
}

void MoistMixtureImpl::_cons2prim(torch::Tensor cons, torch::Tensor &prim) {
  apply_conserved_limiter_(cons);
  int ny = pthermo->options.vapor_ids().size() +
           pthermo->options.cloud_ids().size() - 1;

  // den -> den
  auto out = prim[IDN];
  torch::sum_out(out, cons.narrow(0, ICY, ny), /*dim=*/0);
  out += cons[IDN];

  // den -> mixr
  out = prim.narrow(0, ICY, ny);
  torch::div_out(out, cons.narrow(0, ICY, ny), prim[IDN]);

  // mom -> vel
  out = prim.narrow(0, IVX, 3);
  torch::div_out(out, cons.narrow(0, IVX, 3), prim[IDN]);

  pcoord->vec_raise_(out);

  // KE (TODO: cli, new kernel for this operation)
  _ke.set_((prim.narrow(0, IVX, 3) * cons.narrow(0, IVX, 3)).sum(0));
  _ke *= 0.5;

  auto ivol = pthermo->compute("DY->V", {prim[IDN], prim.narrow(0, ICY, ny)});

  torch::sub_out(_ie, cons[IPR], _ke);

  auto temp = pthermo->compute("VU->T", {ivol, _ie});
  prim[IPR] = pthermo->compute("VT->P", {ivol, temp});

  apply_primitive_limiter_(prim);
}

void MoistMixtureImpl::_cons2ke(torch::Tensor cons, torch::Tensor &out) {
  int ny = pthermo->options.vapor_ids().size() +
           pthermo->options.cloud_ids().size() - 1;
  auto rho = get_buffer("thermo.D");
  rho.set_(cons[IDN] + cons.narrow(0, ICY, ny).sum(0));

  auto mom = cons.narrow(0, IVX, 3).clone();
  pcoord->vec_raise_(mom);
  out.set_(0.5 * (cons.narrow(0, IVX, 3) * mom).sum(0) / rho);
}

void MoistMixtureImpl::_adiabatic_index(torch::Tensor ivol, torch::Tensor temp,
                                        torch::Tensor &out) const {
  auto conc = ivol * pthermo->inv_mu;
  auto cp = kintera::eval_cp_R(temp, conc, pthermo->options);
  auto cv = kintera::eval_cv_R(temp, conc, pthermo->options);

  auto cp_vol = (conc * cp).sum(-1);
  auto cv_vol = (conc * cv).sum(-1);
  torch::div_out(out, cp_vol, cv_vol);
}

void MoistMixtureImpl::_isothermal_sound_speed(torch::Tensor ivol,
                                               torch::Tensor temp,
                                               torch::Tensor dens,
                                               torch::Tensor &out) const {
  int nvapor = pthermo->options.vapor_ids().size();
  auto conc_gas = (ivol * pthermo->inv_mu).narrow(-1, 0, nvapor);
  auto cz = kintera::eval_czh(temp, conc_gas, pthermo->options);
  auto cz_ddC = kintera::eval_czh_ddC(temp, conc_gas, pthermo->options);

  auto result = torch::addcmul(cz, cz_ddC, conc_gas);
  result *= conc_gas;

  out.set_(result.sum(-1));
  out *= kintera::constants::Rgas * temp / dens;
  out.sqrt_();
}

}  // namespace snap
