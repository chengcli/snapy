// kintere
#include <kintera/constants.h>

#include <kintera/thermo/eval_uhs.hpp>
#include <kintera/thermo/thermo.hpp>

// snap
#include <snap/snap.h>

#include <snap/registry.hpp>

#include "ideal_moist.hpp"

namespace snap {

IdealMoistImpl::IdealMoistImpl(EquationOfStateOptions const &options_)
    : EquationOfStateImpl(options_) {
  reset();
}

void IdealMoistImpl::reset() {
  // set up coordinate model
  pcoord = register_module_op(this, "coord", options.coord());

  // set up thermodynamics model
  pthermo = register_module("thermo", kintera::ThermoY(options.thermo()));

  // populate buffers
  int ny = pthermo->options.vapor_ids().size() +
           pthermo->options.cloud_ids().size() - 1;

  inv_mu_ratio_m1 =
      register_buffer("inv_mu_ratio_m1", torch::zeros({ny}, torch::kFloat64));

  for (int i = 0; i < ny; ++i) {
    inv_mu_ratio_m1[i] = pthermo->inv_mu[i + 1] / pthermo->inv_mu[0] - 1.;
  }

  cv_ratio_m1 =
      register_buffer("cv_ratio_m1", torch::zeros({ny}, torch::kFloat64));

  auto Rd = kintera::constants::Rgas * pthermo->inv_mu[0];
  for (int i = 0; i < ny; ++i) {
    auto Ri = kintera::constants::Rgas * pthermo->inv_mu[i + 1];
    cv_ratio_m1[i] = (pthermo->options.cref_R()[1 + i] * Ri) /
                         (pthermo->options.cref_R()[0] * Rd) -
                     1.;
  }

  u0 = register_buffer(
      "u0", torch::tensor(pthermo->options.uref_R(), torch::kFloat64));

  u0 *= kintera::constants::Rgas * pthermo->inv_mu;
}

torch::Tensor IdealMoistImpl::compute(std::string ab,
                                      std::vector<torch::Tensor> const &args) {
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
    auto gammad =
        (pthermo->options.cref_R()[0] + 1) / pthermo->options.cref_R()[0];
    int ny = pthermo->options.vapor_ids().size() +
             pthermo->options.cloud_ids().size() - 1;
    auto feps = f_eps(w.narrow(0, ICY, ny));
    auto fsig = f_sig(w.narrow(0, ICY, ny));
    return 1. + (gammad - 1.) * feps / fsig;
  } else if (ab == "WA->L") {
    auto w = args[0];
    auto gamma = args[1];
    auto dens = w[IDN];
    auto pres = w[IPR];
    return torch::sqrt(gamma * pres / dens);
  } else {
    TORCH_CHECK(false, "Unknown abbreviation: ", ab);
  }
}

void IdealMoistImpl::_prim2cons(torch::Tensor prim, torch::Tensor &cons) {
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

void IdealMoistImpl::_cons2prim(torch::Tensor cons, torch::Tensor &prim) {
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

  // subtract the internal energy offset
  ie -= cons[IDN] * u0[0];

  std::vector<int64_t> vec(cons.dim(), 1);
  vec[0] = -1;
  ie -= (cons.narrow(0, ICY, ny) * u0.narrow(0, 1, ny).view(vec)).sum(0);

  // eng -> pr
  auto gammad =
      (pthermo->options.cref_R()[0] + 1) / pthermo->options.cref_R()[0];

  // TODO(cli) iteration needed here
  auto yfrac = prim.narrow(0, ICY, ny);
  prim[IPR] = (gammad - 1) * ie * f_eps(yfrac) / f_sig(yfrac);

  apply_primitive_limiter_(prim);
}

torch::Tensor IdealMoistImpl::_prim2intEng(torch::Tensor prim) {
  int ny = pthermo->options.vapor_ids().size() +
           pthermo->options.cloud_ids().size() - 1;

  auto yfrac = prim.narrow(0, ICY, ny);

  auto gammad =
      (pthermo->options.cref_R()[0] + 1) / pthermo->options.cref_R()[0];

  // TODO(cli) iteration needed here
  auto ie = prim[IPR] * f_sig(yfrac) / f_eps(yfrac) / (gammad - 1);

  // add the internal energy offset
  auto yd = 1. - yfrac.sum(0);
  ie += prim[IDN] * yd * u0[0];
  ie +=
      prim[IDN] * yfrac.unfold(0, ny, 1).matmul(u0.narrow(0, 1, ny)).squeeze(0);
  return ie;
}

torch::Tensor IdealMoistImpl::_prim2temp(torch::Tensor prim) {
  int ny = pthermo->options.vapor_ids().size() +
           pthermo->options.cloud_ids().size() - 1;
  auto Rd = kintera::constants::Rgas / kintera::species_weights[0];
  auto yfrac = prim.narrow(0, ICY, ny);
  return prim[IPR] / (prim[IDN] * Rd * f_eps(yfrac));
}

torch::Tensor IdealMoistImpl::_prim2speciesEng(torch::Tensor prim) {
  int ny = pthermo->options.vapor_ids().size() +
           pthermo->options.cloud_ids().size() - 1;

  auto mud = kintera::species_weights[0];
  auto Rd = kintera::constants::Rgas / mud;
  auto cvd = kintera::species_cref_R[0] * Rd;

  auto yfrac = prim.narrow(0, ICY, ny);
  auto temp = prim[IPR] / (prim[IDN] * Rd * f_eps(yfrac));

  auto rhos = prim[IDN] * yfrac;

  std::vector<int64_t> vec = {ny, 1, 1, 1};
  auto ie = rhos * (u0.narrow(0, 1, ny).view(vec) +
                    (cv_ratio_m1.view(vec) + 1.) * cvd * temp);

  auto vel = prim.narrow(0, IVX, 3).clone();
  pcoord->vec_lower_(vel);
  auto ke = 0.5 * (prim.narrow(0, IVX, 3) * vel).sum(0, /*keepdim=*/true);

  return ie + ke * rhos;
}

torch::Tensor IdealMoistImpl::_cons2ke(torch::Tensor cons) {
  int ny = pthermo->options.vapor_ids().size() +
           pthermo->options.cloud_ids().size() - 1;
  auto rho = cons[IDN] + cons.narrow(0, ICY, ny).sum(0);

  auto mom = cons.narrow(0, IVX, 3).clone();
  pcoord->vec_raise_(mom);
  return 0.5 * (cons.narrow(0, IVX, 3) * mom).sum(0) / rho;
}

torch::Tensor IdealMoistImpl::_temp2intEng(torch::Tensor cons,
                                           torch::Tensor temp) {
  int ny = pthermo->options.vapor_ids().size() +
           pthermo->options.cloud_ids().size() - 1;

  // internal energy offset
  auto ie = cons[IDN] * u0[0];

  std::vector<int64_t> vec(cons.dim(), 1);
  vec[0] = -1;
  ie += (cons.narrow(0, ICY, ny) * u0.narrow(0, 1, ny).view(vec)).sum(0);

  auto mud = kintera::species_weights[0];
  auto Rd = kintera::constants::Rgas / mud;
  auto cvd = kintera::species_cref_R[0] * Rd;
  auto cvy = (cv_ratio_m1 + 1.) * cvd;

  ie += (cons.narrow(0, ICY, ny) * cvy.view(vec)).sum(0);
  ie *= temp;
  return ie;
}

torch::Tensor IdealMoistImpl::f_eps(torch::Tensor const &yfrac) const {
  int nvapor = pthermo->options.vapor_ids().size() - 1;
  int ncloud = pthermo->options.cloud_ids().size();

  if (nvapor == 0 && ncloud == 0) {
    auto vec = yfrac.sizes().vec();
    vec.erase(vec.begin());
    return torch::ones(vec, yfrac.options());
  }

  auto yu = yfrac.narrow(0, 0, nvapor).unfold(0, nvapor, 1);
  return 1. + yu.matmul(inv_mu_ratio_m1.narrow(0, 0, nvapor)).squeeze(0) -
         yfrac.narrow(0, nvapor, ncloud).sum(0);
}

torch::Tensor IdealMoistImpl::f_sig(torch::Tensor const &yfrac) const {
  int ny = pthermo->options.vapor_ids().size() +
           pthermo->options.cloud_ids().size() - 1;

  if (ny == 0) {
    auto vec = yfrac.sizes().vec();
    vec.erase(vec.begin());
    return torch::ones(vec, yfrac.options());
  }

  auto yu = yfrac.unfold(0, ny, 1);
  return 1. + yu.matmul(cv_ratio_m1).squeeze(0);
}

}  // namespace snap
