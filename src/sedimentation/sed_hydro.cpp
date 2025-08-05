// snap
#include <snap/snap.h>

#include <snap/registry.hpp>

#include "sedimentation.hpp"

namespace snap {

void SedHydroImpl::reset() {
  // register submodules
  peos = register_module_op(this, "eos", options.eos());
  psedvel = register_module("sedvel", SedVel(options.sedvel()));

  // register buffer
  vsed = register_buffer("vsed", torch::empty({0}, torch::kFloat64));
  hydro_ids = register_buffer("hydro_ids",
                              torch::tensor(options.hydro_ids(), torch::kLong));
}

torch::Tensor SedHydroImpl::forward(torch::Tensor wr,
                                    torch::optional<torch::Tensor> out) {
  auto flux = out.value_or(torch::zeros_like(wr));

  // null-op
  if (options.sedvel().grav() == 0. || options.sedvel().species().size() == 0) {
    return flux;
  }

  auto vel = wr.narrow(0, IVX, 3).clone();
  peos->pcoord->vec_lower_(vel);

  auto temp = peos->compute("W->T", {wr});
  vsed.set_(psedvel->forward(wr[Index::IDN], wr[Index::IPR], temp));

  // seal top boundary
  int ie = peos->pcoord->ie();
  int ng = vsed.size(-1) - (ie + 1);
  vsed.narrow(-1, ie + 1, ng).fill_(0.);

  // seal bottom
  int is = peos->pcoord->is();
  vsed.slice(-1, 0, is + 1).fill_(0.);

  // 5 is number of hydro variables
  auto en = peos->compute("W->E", {wr}).index_select(0, hydro_ids - 5);

  auto rhos = wr[IDN] * wr.index_select(0, hydro_ids);
  auto rhos_vsed = rhos * vsed;

  flux.index_add_(0, hydro_ids, rhos_vsed);
  flux.narrow(0, IVX, 3) += vel * rhos_vsed.sum(0, /*keepdim=*/true);
  flux[Index::IPR] += (vsed * en).sum(0);

  return flux;
}

}  // namespace snap
