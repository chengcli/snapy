// kintera
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
  ivol = register_buffer("ivol", torch::empty({0}, torch::kFloat64));
  temp = register_buffer("temp", torch::empty({0}, torch::kFloat64));
  w1 = register_buffer("w1", torch::empty({0}, torch::kFloat64));
}

torch::Tensor MoistMixtureImpl::compute(
    std::string ab, std::vector<torch::Tensor> const &args) {
  if (ab == "W->U") {
    auto w = args[0];
    auto u = args.size() > 1 ? args[1] : torch::empty_like(w);
    _prim2cons(w, u);
    return u;
  } else if (ab == "U->W") {
    auto u = args[0];
    auto w = args.size() > 1 ? args[1] : torch::empty_like(u);
    _cons2prim(u, w);
    return w;
  } else if (ab == "W->I") {
    auto w = args[0];
    return _prim2intEng(w);
  } else if (ab == "W->T") {
    auto w = args[0];
    return _prim2temp(w);
  } else if (ab == "W->E") {
    auto w = args[0];
    return _prim2speciesEng(w);
  } else if (ab == "U->K") {
    auto u = args[0];
    return _cons2ke(u);
  } else if (ab == "UT->I") {
    auto u = args[0];
    auto temp = args[1];
    return _temp2intEng(u, temp);
  } else if (ab == "W->A") {
    auto w = args[0];

    if (!_check_copy(w, w1)) {
      int ny = pthermo->options.vapor_ids().size() +
               pthermo->options.cloud_ids().size() - 1;
      ivol.set_(pthermo->compute("DY->V", {w[IDN], w.narrow(0, ICY, ny)}));
      temp.set_(pthermo->compute("PV->T", {w[IPR], ivol}));
    }

    return _adiabatic_index(ivol, temp);
  } else if (ab == "WA->L") {
    auto w = args[0];
    auto gamma = args[1];

    if (!_check_copy(w, w1)) {
      int ny = pthermo->options.vapor_ids().size() +
               pthermo->options.cloud_ids().size() - 1;
      ivol.set_(pthermo->compute("DY->V", {w[IDN], w.narrow(0, ICY, ny)}));
      temp.set_(pthermo->compute("PV->T", {w[IPR], ivol}));
    }

    auto ct = _isothermal_sound_speed(ivol, temp, w[IDN]);
    return gamma.sqrt() * ct;
  } else {
    TORCH_CHECK(false, "Unknown abbreviation: ", ab);
  }
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
  auto ke = 0.5 * (prim.narrow(0, IVX, 3) * cons.narrow(0, IVX, 3)).sum(0);

  // IE
  cons[IPR] = _prim2intEng(prim);
  cons[IPR] += ke;

  apply_conserved_limiter_(cons);
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

  auto ke = 0.5 * (prim.narrow(0, IVX, 3) * cons.narrow(0, IVX, 3)).sum(0);
  auto ie = cons[IPR] - ke;

  ivol.set_(pthermo->compute("DY->V", {prim[IDN], prim.narrow(0, ICY, ny)}));
  temp.set_(pthermo->compute("VU->T", {ivol, ie}));
  prim[IPR] = pthermo->compute("VT->P", {ivol, temp});
  w1.set_(prim.clone());

  apply_primitive_limiter_(prim);
}

torch::Tensor MoistMixtureImpl::_prim2intEng(torch::Tensor prim) {
  if (!_check_copy(prim, w1)) {
    int ny = pthermo->options.vapor_ids().size() +
             pthermo->options.cloud_ids().size() - 1;
    ivol.set_(pthermo->compute("DY->V", {prim[IDN], prim.narrow(0, ICY, ny)}));
    temp.set_(pthermo->compute("PV->T", {prim[IPR], ivol}));
  }
  return pthermo->compute("VT->U", {ivol, temp});
}

torch::Tensor MoistMixtureImpl::_prim2temp(torch::Tensor prim) {
  if (!_check_copy(prim, w1)) {
    int ny = pthermo->options.vapor_ids().size() +
             pthermo->options.cloud_ids().size() - 1;
    auto yfrac = prim.narrow(0, ICY, ny);
    ivol.set_(pthermo->compute("DY->V", {prim[IDN], yfrac}));
    temp.set_(pthermo->compute("PV->T", {prim[IPR], ivol}));
  }

  return temp;
}

torch::Tensor MoistMixtureImpl::_prim2speciesEng(torch::Tensor prim) {
  int ny = pthermo->options.vapor_ids().size() +
           pthermo->options.cloud_ids().size() - 1;

  auto yfrac = prim.narrow(0, ICY, ny);

  if (!_check_copy(prim, w1)) {
    ivol.set_(pthermo->compute("DY->V", {prim[IDN], yfrac}));
    temp.set_(pthermo->compute("PV->T", {prim[IPR], ivol}));
  }

  auto Rgas = kintera::constants::Rgas * pthermo->inv_mu;
  auto ie = eval_intEng_R(temp, ivol, pthermo->options) * Rgas * ivol;

  auto vel = prim.narrow(0, IVX, 3).clone();

  pcoord->vec_lower_(vel);
  auto ke = 0.5 * (prim.narrow(0, IVX, 3) * vel).sum(0, /*keepdim=*/true);

  auto rhoc = prim[IDN] * yfrac;
  return ie.narrow(-1, 1, ny).permute({3, 0, 1, 2}) + ke * rhoc;
}

torch::Tensor MoistMixtureImpl::_cons2ke(torch::Tensor cons) {
  int ny = pthermo->options.vapor_ids().size() +
           pthermo->options.cloud_ids().size() - 1;
  auto rho = cons[IDN] + cons.narrow(0, ICY, ny).sum(0);
  auto mom = cons.narrow(0, IVX, 3).clone();
  pcoord->vec_raise_(mom);
  return 0.5 * (cons.narrow(0, IVX, 3) * mom).sum(0) / rho;
}

torch::Tensor MoistMixtureImpl::_temp2intEng(torch::Tensor cons,
                                             torch::Tensor T) {
  int ny = pthermo->options.vapor_ids().size() +
           pthermo->options.cloud_ids().size() - 1;
  auto vec = T.sizes().vec();
  vec.push_back(ny + 1);

  auto V = torch::empty(vec, T.options());
  V.select(-1, IDN) = cons[IDN];
  V.narrow(-1, 1, ny) = cons.narrow(0, ICY, ny).permute({1, 2, 3, 0});
  return pthermo->compute("VT->U", {V, T});
}

torch::Tensor MoistMixtureImpl::_adiabatic_index(torch::Tensor V,
                                                 torch::Tensor T) {
  auto conc = V * pthermo->inv_mu;
  auto cp = kintera::eval_cp_R(T, conc, pthermo->options);
  auto cv = kintera::eval_cv_R(T, conc, pthermo->options);

  auto cp_vol = (conc * cp).sum(-1);
  auto cv_vol = (conc * cv).sum(-1);
  return cp_vol / cv_vol;
}

torch::Tensor MoistMixtureImpl::_isothermal_sound_speed(torch::Tensor V,
                                                        torch::Tensor T,
                                                        torch::Tensor dens) {
  int nvapor = pthermo->options.vapor_ids().size();
  auto conc_gas = (V * pthermo->inv_mu).narrow(-1, 0, nvapor);
  auto cz = kintera::eval_czh(T, conc_gas, pthermo->options);
  auto cz_ddC = kintera::eval_czh_ddC(T, conc_gas, pthermo->options);

  auto result = torch::addcmul(cz, cz_ddC, conc_gas);
  result *= conc_gas;

  auto ct = result.sum(-1);
  ct *= kintera::constants::Rgas * T / dens;
  ct.sqrt_();

  return ct;
}

bool MoistMixtureImpl::_check_copy(torch::Tensor prim,
                                   torch::Tensor prim_cache) const {
  if (prim.dim() != prim_cache.dim() || prim.sizes() != prim_cache.sizes()) {
    prim_cache.resize_as_(prim);
    prim_cache.copy_(prim);
    return false;
  }

  if ((prim - prim_cache).abs().max().item<double>() < 1e-10) {
    return true;
  } else {
    prim_cache.copy_(prim);
    return false;
  }
}

}  // namespace snap
