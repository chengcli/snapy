// base
#include <configure.h>

// snap
#include <snap/snap.h>

#include "hydro_formatter.hpp"
#include "primitive_projector.hpp"

namespace snap {
PrimitiveProjectorImpl::PrimitiveProjectorImpl(
    PrimitiveProjectorOptions options_)
    : options(options_) {
  reset();
}

void PrimitiveProjectorImpl::reset() {
  // set up thermodynamic model
  pthermo = register_module("thermo", Thermodynamics(options.thermo()));
}

torch::Tensor PrimitiveProjectorImpl::forward(torch::Tensor w,
                                              torch::Tensor dz) {
  torch::NoGradGuard no_grad;

  if (options.type() == "none") {
    return w;
  }

  int is = options.nghost();
  int ie = w.size(3) - options.nghost();
  SET_SHARED("hydro/psf") =
      calc_hydrostatic_pressure(w, -options.grav(), dz, is, ie);

  auto result = w.clone();

  result[Index::IPR] = calc_nonhydrostatic_pressure(
      w[Index::IPR], GET_SHARED("hydro/psf"), options.margin());

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
  wlr.select(1, Index::IPR).slice(3, is, ie + 1) +=
      GET_SHARED("hydro/psf").slice(2, is, ie + 1);

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
