// yaml
#include <yaml-cpp/yaml.h>

// kintera
#include <kintera/constants.h>

#include <kintera/species.hpp>

// snap
#include <snap/snap.h>

#include "primitive_projector.hpp"

namespace snap {

PrimitiveProjectorOptions PrimitiveProjectorOptions::from_yaml(
    YAML::Node const &root) {
  PrimitiveProjectorOptions op;

  if (!root["dynamics"]) return op;

  if (root["dynamics"]["vertical-projection"]) {
    op.type() =
        root["dynamics"]["vertical-projection"]["type"].as<std::string>("none");

    op.margin() =
        root["dynamics"]["vertical-projection"]["pressure-margin"].as<double>(
            1.e-6);
  }

  if (root["forcing"]) {
    if (root["forcing"]["const-gravity"])
      op.grav() = root["forcing"]["const-gravity"]["grav1"].as<double>(0.);
  }

  if (kintera::species_weights.size() == 0) {
    TORCH_CHECK(false,
                "PrimitiveProjectorOptions: species is not initialized. ",
                "Please initialize it first.");
  }

  auto mu = kintera::species_weights[0];
  op.Rd() = kintera::constants::Rgas / mu;

  if (!root["geometry"]) return op;
  if (!root["geometry"]["cells"]) return op;

  op.nghost() = root["geometry"]["cells"]["nghost"].as<int>(1);

  return op;
}

PrimitiveProjectorImpl::PrimitiveProjectorImpl(
    PrimitiveProjectorOptions options_)
    : options(options_) {
  reset();
}

void PrimitiveProjectorImpl::reset() {
  // populate buffer
  _psf = register_buffer("psf", torch::empty({0}, torch::kFloat64));
}

torch::Tensor PrimitiveProjectorImpl::forward(torch::Tensor w,
                                              torch::Tensor dz) {
  if (options.type() == "none") {
    return w;
  }

  int is = options.nghost();
  int ie = w.size(3) - options.nghost();
  _psf.set_(calc_hydrostatic_pressure(w, -options.grav(), dz, is, ie));

  auto result = w.clone();

  result[Index::IPR] =
      calc_nonhydrostatic_pressure(w[Index::IPR], _psf, options.margin());

  if (options.type() == "temperature") {
    result[Index::IDN] = w[Index::IPR] / (w[Index::IDN] * options.Rd());
  } else if (options.type() == "density") {
    // do nothing
  } else {
    throw std::runtime_error("Unknown primitive projector type: " +
                             options.type());
  }

  return result;
}

void PrimitiveProjectorImpl::restore_inplace(torch::Tensor wlr) {
  if (options.type() == "none") {
    return;
  }

  int is = options.nghost();
  int ie = wlr.size(4) - options.nghost();

  // restore pressure
  wlr.select(1, Index::IPR).slice(3, is, ie + 1) += _psf.slice(2, is, ie + 1);

  // restore density
  if (options.type() == "temperature") {
    wlr.select(1, Index::IDN).slice(3, is, ie + 1) =
        wlr.select(1, Index::IPR).slice(3, is, ie + 1) /
        (wlr.select(1, Index::IDN).slice(3, is, ie + 1) * options.Rd());
  } else if (options.type() == "density") {
    // do nothing
  } else {
    throw std::runtime_error("Unknown primitive projector type: " +
                             options.type());
  }
}

torch::Tensor calc_hydrostatic_pressure(torch::Tensor w, double grav,
                                        torch::Tensor dz, int is, int ie) {
  auto psf = torch::zeros({w.size(1), w.size(2), w.size(3) + 1}, w.options());
  auto nc1 = w.size(3);

  // lower ghost zones and interior
  psf.slice(2, 0, ie) =
      grav * w[Index::IDN].slice(2, 0, ie) * dz.slice(0, 0, ie);

  // flip lower ghost zones
  psf.slice(2, 0, is) *= -1;

  // isothermal extrapolation to top boundary
  auto RdTv = w[Index::IPR].select(2, ie - 1) / w[Index::IDN].select(2, ie - 1);
  psf.select(2, ie) =
      w[Index::IPR].select(2, ie - 1) * exp(-grav * dz[ie - 1] / (2. * RdTv));

  // upper ghost zones
  psf.slice(2, ie + 1, nc1 + 1) =
      grav * w[Index::IDN].slice(2, ie, nc1) * dz.slice(0, ie, nc1);

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
