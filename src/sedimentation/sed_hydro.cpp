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
  vsed = register_buffer("vsed", torch::empty({0}));
}

torch::Tensor SedHydroImpl::forward(torch::Tensor hydro_w,
                                    torch::optional<torch::Tensor> out) {
  auto flux = out.value_or(torch::zeros_like(hydro_w));

  int ncloud = options.sedvel().radius().size();
  int nvapor = hydro_w.size(0) - 5 - ncloud;  // 5 = IDN, IPR, IVX, IVY, IVZ

  // null-op
  if (options.sedvel().grav() == 0. || ncloud == 0) {
    return flux;
  }

  auto vel = hydro_w.narrow(0, IVX, 3).clone();
  peos->pcoord->vec_lower_(vel);

  auto temp = peos->compute("W->T", {hydro_w});
  vsed.set_(psedvel->forward(hydro_w[Index::IDN], hydro_w[Index::IPR], temp));

  auto en = peos->compute("W->E", {hydro_w});
  auto rhoc = peos->get_buffer("C");

  flux.narrow(0, Index::ICY + nvapor, ncloud) += rhoc * vsed;
  flux.narrow(0, IVX, 3) += vel * (rhoc * vsed).sum(0, /*keepdim=*/true);
  flux[Index::IPR] += (rhoc * vsed * en).sum(0);

  return flux;
}

}  // namespace snap
