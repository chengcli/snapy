// yaml
#include <yaml-cpp/yaml.h>

// kintera
#include <kintera/constants.h>

#include <kintera/species.hpp>

// snap
#include <snap/snap.h>

#include <snap/coord/coordinate.hpp>
#include <snap/eos/equation_of_state.hpp>
#include <snap/forcing/forcing.hpp>
#include <snap/hydro/hydro.hpp>
#include <snap/mesh/meshblock.hpp>

#include "primitive_projector.hpp"

namespace snap {

PrimitiveProjectorOptions PrimitiveProjectorOptionsImpl::from_yaml(
    std::string const &filename, bool verbose) {
  auto config = YAML::LoadFile(filename);
  if (!config["dynamics"]) return nullptr;
  if (!config["dynamics"]["vertical-projection"]) return nullptr;
  return from_yaml(config["dynamics"]["vertical-projection"]);
}

PrimitiveProjectorOptions PrimitiveProjectorOptionsImpl::from_yaml(
    YAML::Node const &node) {
  auto op = PrimitiveProjectorOptionsImpl::create();

  op->type() = node["type"].as<std::string>("none");
  op->margin() = node["pressure-margin"].as<double>(1.e-6);

  return op;
}

PrimitiveProjectorImpl::PrimitiveProjectorImpl(
    PrimitiveProjectorOptions options_, torch::nn::Module *p)
    : options(options_) {
  phydro = dynamic_cast<HydroImpl const *>(p);
  reset();
}

void PrimitiveProjectorImpl::reset() {
  TORCH_CHECK(phydro, "[PrimitiveProjector] Parent Hydro is null");

  // populate buffer
  _psf = register_buffer("psf", torch::empty({0}, torch::kFloat64));
}

torch::Tensor PrimitiveProjectorImpl::forward(torch::Tensor w,
                                              torch::Tensor dz) {
  if (options->type() == "none") {
    return w;
  }

  auto pcoord = phydro->pmb->pcoord;

  int is = pcoord->il();
  int ie = pcoord->iu() + 1;
  auto grav = -phydro->options->grav()->grav1();
  _psf.set_(calc_hydrostatic_pressure(w, grav, dz, is, ie));

  auto result = w.clone();

  result[IPR] = calc_nonhydrostatic_pressure(w[IPR], _psf, options->margin());

  if (options->type() == "temperature") {
    auto Rd = kintera::constants::Rgas / phydro->peos->species_weight();
    result[IDN] = w[IPR] / (w[IDN] * Rd);
  } else if (options->type() == "density") {
    // do nothing
  } else {
    throw std::runtime_error("Unknown primitive projector type: " +
                             options->type());
  }

  return result;
}

void PrimitiveProjectorImpl::restore_inplace(torch::Tensor wlr) {
  if (options->type() == "none") {
    return;
  }

  auto pcoord = phydro->pmb->pcoord;

  int is = pcoord->il();
  int ie = pcoord->iu() + 1;

  // restore pressure
  wlr.select(1, IPR).slice(3, is, ie + 1) += _psf.slice(2, is, ie + 1);

  // restore density
  if (options->type() == "temperature") {
    auto Rd = kintera::constants::Rgas / phydro->peos->species_weight();
    wlr.select(1, IDN).slice(3, is, ie + 1) =
        wlr.select(1, IPR).slice(3, is, ie + 1) /
        (wlr.select(1, IDN).slice(3, is, ie + 1) * Rd);
  } else if (options->type() == "density") {
    // do nothing
  } else {
    throw std::runtime_error("Unknown primitive projector type: " +
                             options->type());
  }
}

std::shared_ptr<PrimitiveProjectorImpl> PrimitiveProjectorImpl::create(
    PrimitiveProjectorOptions const &opts, torch::nn::Module *p,
    std::string const &name) {
  TORCH_CHECK(p != nullptr, "[PrimitiveProjector] Parent module is nullptr");
  TORCH_CHECK(opts != nullptr,
              "[PrimitiveProjector] Options pointer is nullptr");

  return p->register_module(name, PrimitiveProjector(opts, p));
}

torch::Tensor calc_hydrostatic_pressure(torch::Tensor w, double grav,
                                        torch::Tensor dz, int is, int ie) {
  auto psf = torch::zeros({w.size(1), w.size(2), w.size(3) + 1}, w.options());
  auto nc1 = w.size(3);

  // lower ghost zones and interior
  psf.slice(2, 0, ie) = grav * w[IDN].slice(2, 0, ie) * dz.slice(0, 0, ie);

  // flip lower ghost zones
  psf.slice(2, 0, is) *= -1;

  // isothermal extrapolation to top boundary
  auto RdTv = w[IPR].select(2, ie - 1) / w[IDN].select(2, ie - 1);
  psf.select(2, ie) =
      w[IPR].select(2, ie - 1) * exp(-grav * dz[ie - 1] / (2. * RdTv));

  // upper ghost zones
  psf.slice(2, ie + 1, nc1 + 1) =
      grav * w[IDN].slice(2, ie, nc1) * dz.slice(0, ie, nc1);

  // integrate downwards
  psf.slice(2, 0, ie + 1) =
      torch::cumsum(psf.slice(2, 0, ie + 1).flip(2), 2).flip(2);

  // integrate upwards
  psf.slice(2, ie, nc1 + 1) = torch::cumsum(psf.slice(2, ie, nc1 + 1), 2);

  return psf;
}

torch::Tensor calc_nonhydrostatic_pressure(torch::Tensor pres,
                                           torch::Tensor psf, double margin) {
  auto nc1 = psf.size(2);
  auto df = psf.slice(2, 0, -1) - psf.slice(2, 1, nc1);
  auto psv = torch::where(df.abs() < margin,
                          0.5 * (psf.slice(2, 0, -1) + psf.slice(2, 1, nc1)),
                          df / log(psf.slice(2, 0, -1) / psf.slice(2, 1, nc1)));
  return pres - psv;
}

}  // namespace snap
