// base
#include <configure.h>

// snap
#include <snap/constants.h>
#include <snap/snap.h>

#include "thermo_formatter.hpp"
#include "thermodynamics.hpp"

namespace snap {
ThermodynamicsOptions::ThermodynamicsOptions(ParameterInput pin) {
  gammad_ref(pin->GetOrAddReal("thermodynamics", "gammad_ref", 1.4));
  nvapor(pin->GetOrAddInteger("thermodynamics", "nvapor", 0));
  ncloud(pin->GetOrAddInteger("thermodynamics", "ncloud", 0));
}

ThermodynamicsImpl::ThermodynamicsImpl(const ThermodynamicsOptions& options_)
    : options(options_) {
  if (options.mu_ratio_m1().empty()) {
    options.mu_ratio_m1() =
        std::vector<double>(options.nvapor() + options.ncloud(), 0.);
  }

  if (options.cv_ratio_m1().empty()) {
    options.cv_ratio_m1() =
        std::vector<double>(options.nvapor() + options.ncloud(), 0.);
  }

  if (options.cp_ratio_m1().empty()) {
    options.cp_ratio_m1() =
        std::vector<double>(options.nvapor() + options.ncloud(), 0.);
  }

  if (options.h0().empty()) {
    options.h0() =
        std::vector<double>(1 + options.nvapor() + options.ncloud(), 0.);
  }

  reset();
}

void ThermodynamicsImpl::reset() {
  TORCH_CHECK(
      options.mu_ratio_m1().size() == options.nvapor() + options.ncloud(),
      "mu_ratio_m1 size mismatch");
  TORCH_CHECK(
      options.cv_ratio_m1().size() == options.nvapor() + options.ncloud(),
      "cv_ratio_m1 size mismatch");
  TORCH_CHECK(
      options.cp_ratio_m1().size() == options.nvapor() + options.ncloud(),
      "cp_ratio_m1 size mismatch");
  TORCH_CHECK(options.h0().size() == 1 + options.nvapor() + options.ncloud(),
              "h0 size mismatch");

  mu_ratio_m1 = register_buffer(
      "mu_ratio_m1", torch::tensor(options.mu_ratio_m1(), torch::kFloat64));

  cv_ratio_m1 = register_buffer(
      "cv_ratio_m1", torch::tensor(options.cv_ratio_m1(), torch::kFloat64));

  cp_ratio_m1 = register_buffer(
      "cp_ratio_m1", torch::tensor(options.cp_ratio_m1(), torch::kFloat64));

  h0 = register_buffer("h0", torch::tensor(options.h0(), torch::kFloat64));

  options.cond().species(options.species());
  pcond = register_module("cond", Condensation(options.cond()));
  options.cond() = pcond->options;
}

int ThermodynamicsImpl::species_index(std::string const& name) const {
  auto it = std::find(options.species().begin(), options.species().end(), name);
  if (it == options.species().end()) {
    throw std::runtime_error("species not found");
  }
  return Index::ICY + std::distance(options.species().begin(), it);
}

torch::Tensor ThermodynamicsImpl::get_gammad(torch::Tensor var,
                                             int type) const {
  if (type == kPrimitive) {
    return torch::ones_like(var[0]) * options.gammad_ref();
  } else if (type == kConserved) {
    return torch::ones_like(var[0]) * options.gammad_ref();
  } else {
    std::stringstream msg;
    msg << fmt::format("{}::Unknown variable type code: {}", name(), type);
    throw std::runtime_error(msg.str());
  }
}

torch::Tensor ThermodynamicsImpl::f_eps(torch::Tensor w) const {
  auto nvapor = options.nvapor();
  auto ncloud = options.ncloud();

  auto wu = w.narrow(0, Index::ICY, nvapor).unfold(0, nvapor, 1);
  return 1. + wu.matmul(mu_ratio_m1.narrow(0, 0, nvapor)).squeeze(0) -
         w.narrow(0, Index::ICY + nvapor, ncloud).sum(0);
}

torch::Tensor ThermodynamicsImpl::f_sig(torch::Tensor w) const {
  auto nvapor = options.nvapor();
  auto ncloud = options.ncloud();

  auto wu =
      w.narrow(0, Index::ICY, nvapor + ncloud).unfold(0, nvapor + ncloud, 1);
  return 1. + wu.matmul(cv_ratio_m1).squeeze(0);
}

torch::Tensor ThermodynamicsImpl::f_psi(torch::Tensor w) const {
  auto nvapor = options.nvapor();
  auto ncloud = options.ncloud();

  auto wu =
      w.narrow(0, Index::ICY, nvapor + ncloud).unfold(0, nvapor + ncloud, 1);
  return 1. + wu.matmul(cp_ratio_m1).squeeze(0);
}

torch::Tensor ThermodynamicsImpl::get_mu() const {
  auto nvapor = options.nvapor();
  auto ncloud = options.ncloud();

  auto result = torch::ones({1 + nvapor + ncloud}, mu_ratio_m1.options());
  result[0] = constants::Rgas / options.Rd();
  result.narrow(0, 1, nvapor + ncloud) = result[0] / (mu_ratio_m1 + 1.);

  return result;
}

torch::Tensor ThermodynamicsImpl::get_cv_ref() const {
  auto nvapor = options.nvapor();
  auto ncloud = options.ncloud();

  auto result = torch::empty({1 + nvapor + ncloud}, cv_ratio_m1.options());
  result[0] = options.Rd() / (options.gammad_ref() - 1.);
  result.narrow(0, 1, nvapor + ncloud) = result[0] * (cv_ratio_m1 + 1.);

  return result;
}

torch::Tensor ThermodynamicsImpl::get_cp_ref() const {
  auto nvapor = options.nvapor();
  auto ncloud = options.ncloud();

  auto result = torch::empty({1 + nvapor + ncloud}, cp_ratio_m1.options());
  result[0] = options.gammad_ref() / (options.gammad_ref() - 1.) * options.Rd();
  result.narrow(0, 1, nvapor + ncloud) = result[0] * (cp_ratio_m1 + 1.);

  return result;
}

torch::Tensor ThermodynamicsImpl::get_temp(torch::Tensor w) const {
  return w[Index::IPR] / (w[Index::IDN] * options.Rd() * f_eps(w));
}

torch::Tensor ThermodynamicsImpl::get_theta_ref(torch::Tensor w,
                                                double p0) const {
  return get_temp(w) * torch::pow(p0 / w[Index::IPR], _get_chi_ref(w));
}

torch::Tensor ThermodynamicsImpl::get_mole_fraction(torch::Tensor yfrac) const {
  int nmass = yfrac.size(0);
  int nc3 = yfrac.size(1);
  int nc2 = yfrac.size(2);
  int nc1 = yfrac.size(3);
  auto xfrac = torch::empty({nc3, nc2, nc1, 1 + nmass}, yfrac.options());

  xfrac.narrow(-1, 1, nmass) = yfrac.permute({1, 2, 3, 0}) * (mu_ratio_m1 + 1.);
  auto sum = 1. + yfrac.permute({1, 2, 3, 0}).matmul(mu_ratio_m1);
  xfrac.narrow(-1, 1, nmass) /= sum.unsqueeze(-1);
  xfrac.select(-1, 0) = 1. - xfrac.narrow(-1, 1, nmass).sum(-1);
  return xfrac;
}

torch::Tensor ThermodynamicsImpl::get_mass_fraction(torch::Tensor xfrac) const {
  int nc3 = xfrac.size(0);
  int nc2 = xfrac.size(1);
  int nc1 = xfrac.size(2);
  int nmass = xfrac.size(3) - 1;
  auto yfrac = torch::empty({nmass, nc3, nc2, nc1}, xfrac.options());

  yfrac.permute({1, 2, 3, 0}) = xfrac.narrow(-1, 1, nmass) / (mu_ratio_m1 + 1.);
  auto sum =
      1. - xfrac.narrow(-1, 1, nmass).matmul(mu_ratio_m1 / (mu_ratio_m1 + 1.));
  return yfrac / sum.unsqueeze(0);
}

torch::Tensor ThermodynamicsImpl::forward(torch::Tensor u) {
  int nvapor = options.nvapor();
  int ncloud = options.ncloud();

  // total density
  auto rho = u[Index::IDN].clone();
  rho += u.narrow(0, Index::ICY, nvapor + ncloud).sum(0);

  auto feps = f_eps(u / rho);
  auto fsig = f_sig(u / rho);

  // auto w = pcoord->vec_raise(u, m);
  auto vel = u.narrow(0, Index::IVX, 3);
  auto ke = 0.5 *
            (u[Index::IVX] * vel[0] + u[Index::IVY] * vel[1] +
             u[Index::IVZ] * vel[2]) /
            rho;

  auto pres = (options.gammad_ref() - 1.) * (u[Index::IPR] - ke) * feps / fsig;
  auto temp = pres / (rho * options.Rd() * feps);
  auto conc = _get_mole_concentration(u);

  // std::cout << "pres = " << pres << std::endl;
  // std::cout << "temp = " << temp << std::endl;

  auto mu = get_mu();
  auto krate = torch::ones_like(temp);

  int iter = 0;
  for (; iter < options.max_iter(); ++iter) {
    /*std::cout << "iter = " << iter << std::endl;
    std::cout << "conc = " << conc << std::endl;*/
    auto intEng_RT = _get_internal_energy_RT_ref(temp);
    /*std::cout << "total intEng = "
              << (conc * intEng_RT).sum(-1) * constants::Rgas * temp
              << std::endl;*/

    auto cv_R = get_cv_ref() * mu / constants::Rgas;

    auto rates = pcond->forward(temp, pres, conc, intEng_RT, cv_R, krate);
    // std::cout << "rates = " << rates << std::endl;
    // std::cout << "krate = " << krate << std::endl;

    conc += rates;
    TORCH_CHECK(conc.min().item<double>() >= 0., "negative concentration");

    auto dT = -temp * (rates * intEng_RT).sum(-1) / conc.matmul(cv_R);
    krate = where(dT * krate > 0., 2 * krate, torch::sign(dT));
    krate.clamp_(-options.boost(), options.boost());

    if (dT.abs().max().item<double>() < options.ftol()) break;
    temp += dT;
    pres = conc.narrow(-1, 0, 1 + nvapor).sum(-1) * constants::Rgas * temp;

    // std::cout << "temp = " << temp << std::endl;
  }

  auto u0 = u.clone();

  u.narrow(0, Index::ICY, nvapor + ncloud) =
      (conc * mu).narrow(-1, 1, nvapor + ncloud).permute({3, 0, 1, 2});

  // total energy
  feps = f_eps(u / rho);
  fsig = f_sig(u / rho);
  u[Index::IPR] = pres * fsig / feps / (options.gammad_ref() - 1.) + ke;

  return u - u0;
}

torch::Tensor ThermodynamicsImpl::equilibrate_tp(torch::Tensor temp,
                                                 torch::Tensor pres,
                                                 torch::Tensor yfrac) const {
  auto xfrac = get_mole_fraction(yfrac);

  int iter = 0;
  for (; iter < options.max_iter(); ++iter) {
    auto rates = pcond->equilibrate_tp(temp, pres, xfrac, 1 + options.nvapor());
    xfrac += rates;

    if ((rates / (xfrac + 1.e-10)).max().item<double>() < options.rtol()) break;
    TORCH_CHECK(xfrac.min().item<double>() >= 0., "negative mole fraction");
  }

  return get_mass_fraction(xfrac) - yfrac;
}

torch::Tensor ThermodynamicsImpl::_get_chi_ref(torch::Tensor w) const {
  auto gammad = options.gammad_ref();
  return (gammad - 1.) / gammad * f_eps(w) / f_psi(w);
}

torch::Tensor ThermodynamicsImpl::_get_mole_concentration(
    torch::Tensor u) const {
  auto nvapor = options.nvapor();
  auto ncloud = options.ncloud();

  auto vec = u.sizes().vec();
  vec.erase(vec.begin());
  vec.push_back(1 + nvapor + ncloud);

  auto result = torch::empty(vec, u.options());
  auto mu = get_mu();

  result.select(3, 0) = u[Index::IDN] / mu[0];
  result.narrow(3, 1, nvapor + ncloud) =
      u.narrow(0, Index::ICY, nvapor + ncloud).permute({1, 2, 3, 0}) /
      mu.narrow(0, 1, nvapor + ncloud);
  return result;
}

torch::Tensor ThermodynamicsImpl::_get_internal_energy_RT_ref(
    torch::Tensor temp) const {
  auto nvapor = options.nvapor();
  auto ncloud = options.ncloud();

  auto vec = temp.sizes().vec();
  vec.push_back(1 + nvapor + ncloud);

  auto mu = get_mu();
  auto cp = get_cp_ref().view({1, 1, 1, -1}) * mu;
  auto result = h0 * mu + cp * (temp - options.Tref()).unsqueeze(3);
  result.narrow(3, 0, 1 + nvapor) -= constants::Rgas * temp;
  return result / (constants::Rgas * temp.unsqueeze(3));
}

}  // namespace snap
