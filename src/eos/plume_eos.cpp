// snap
#include "plume_eos.hpp"

#include <snap/snap.h>

namespace snap {

PlumeEOSImpl::PlumeEOSImpl(EquationOfStateOptions const &options_)
    : EquationOfStateImpl(options_) {
  reset();
}

void PlumeEOSImpl::reset() {
  pcoord = CoordinateImpl::create(options->coord(), this);
}

torch::Tensor PlumeEOSImpl::compute(std::string ab,
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
  } else if (ab == "W->T") {
    auto w = args[0];
    return torch::zeros_like(w[IDN]);
  } else if (ab == "W->A") {
    auto w = args[0];
    return torch::ones_like(w[IDN]);
  } else if (ab == "W->L") {
    auto w = args[0];
    return torch::max(w[IVX].abs(), w[IVZ].abs());
  } else {
    TORCH_CHECK(false, "Unknown abbreviation: ", ab);
  }
}

void PlumeEOSImpl::_prim2cons(torch::Tensor prim, torch::Tensor &cons) {
  cons[IDN] = prim[IDN] * prim[IDN];
  cons[IVX] = prim[IDN] * prim[IDN] * prim[IVX];
  cons[IVY] = prim[IDN] * prim[IDN] * prim[IVY];
  cons[IVZ] = prim[IDN] * prim[IDN] * prim[IDN] * prim[IVZ];
}

void PlumeEOSImpl::_cons2prim(torch::Tensor cons, torch::Tensor &prim) {
  prim[IDN] = torch::sqrt(cons[IDN]);
  prim[IVX] = cons[IVX] / cons[IDN];
  prim[IVY] = cons[IVY] / cons[IDN];
  prim[IVZ] = cons[IVZ] / (cons[IDN] * prim[IDN]);
}

}  // namespace snap
