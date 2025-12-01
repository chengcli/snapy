// snap
#include "interpolation.hpp"

namespace snap {

Interp InterpImpl::create(InterpOptions const& opts, torch::nn::Module* p,
                          std::string const& name) {
  TORCH_CHECK(opts, "Interp options is null");
  TORCH_CHECK(p, "Parent module is null");

  if (opts->type() == "dc") {
    return p->register_module(name, DonorCellInterp(opts));
  } else if (opts->type() == "plm") {
    return p->register_module(name, PLMInterp(opts));
  } else if (opts->type() == "ppm") {
    return p->register_module(name, PPMInterp(opts));
  } else if (opts->type() == "cp3") {
    return p->register_module(name, Center3Interp(opts));
  } else if (opts->type() == "cp5") {
    return p->register_module(name, Center5Interp(opts));
  } else if (opts->type() == "weno3") {
    if (name.back() == '1') {
      return p->register_module(name, Weno3Interp(opts));
    } else if (name.back() == '2') {
      return p->register_module(name, Center3Interp(opts));
    } else {
      TORCH_CHECK(false, "Interp: unknown name " + name);
    }
  } else if (opts->type() == "weno5") {
    if (name.back() == '1') {
      return p->register_module(name, Weno5Interp(opts));
    } else if (name.back() == '2') {
      return p->register_module(name, Center5Interp(opts));
    } else {
      TORCH_CHECK(false, "Interp: unknown name " + name);
    }
  } else {
    TORCH_CHECK(false, "Interp: unknown type " + opts->type());
  }
}

}  // namespace snap
