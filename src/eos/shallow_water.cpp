// snap
#include "shallow_water.hpp"

#include <snap/snap.h>

#include <snap/coord/coord_utils.hpp>
#include <snap/hydro/hydro.hpp>
#include <snap/mesh/meshblock.hpp>

namespace snap {

void ShallowWaterImpl::reset() {}

torch::Tensor ShallowWaterImpl::compute(
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
  } else if (ab == "W->A") {
    return torch::Tensor();
  } else if (ab == "W->T") {
    return torch::Tensor();
  } else if (ab == "WA->L") {
    auto w = args[0];
    return torch::sqrt(w[IDN]);
  } else {
    TORCH_CHECK(false, "Unknown abbreviation: ", ab);
  }
}

void ShallowWaterImpl::_cons2prim(torch::Tensor cons, torch::Tensor &prim) {
  auto pcoord = phydro->pmb->pcoord;
  apply_conserved_limiter_(cons);

  prim[IDN] = cons[IDN];

  // lvalue view
  auto out = prim.narrow(0, IVX, 3);
  torch::div_out(out, cons.narrow(0, IVX, 3), cons[IDN]);

  coord_vec_raise_(out, pcoord->cosine_cell_kj);

  apply_primitive_limiter_(prim);
}

void ShallowWaterImpl::_prim2cons(torch::Tensor prim, torch::Tensor &cons) {
  auto pcoord = phydro->pmb->pcoord;
  apply_primitive_limiter_(prim);

  cons[IDN] = prim[IDN];

  // lvalue view
  auto out = cons.narrow(0, IVX, 3);
  torch::mul_out(out, prim.narrow(0, IVX, 3), prim[IDN]);

  coord_vec_lower_(out, pcoord->cosine_cell_kj);

  apply_conserved_limiter_(cons);
}

}  // namespace snap
