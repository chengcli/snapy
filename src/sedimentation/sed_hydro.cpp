// snap
#include <snap/snap.h>

#include <snap/coord/coord_utils.hpp>
#include <snap/hydro/hydro.hpp>

#include "sedimentation.hpp"

namespace snap {

SedHydroImpl::SedHydroImpl(SedHydroOptions const& options_,
                           torch::nn::Module* p)
    : options(options_) {
  phydro = dynamic_cast<HydroImpl const*>(p);
  reset();
}

void SedHydroImpl::reset() {
  TORCH_CHECK(phydro, "Parent Hydro module is nullptr");

  psedvel = SedVelImpl::create(options->sedvel(), this);

  // register buffer
  vsed = register_buffer("vsed", torch::empty({0}, torch::kFloat64));
  hydro_ids = register_buffer(
      "hydro_ids", torch::tensor(options->hydro_ids(), torch::kLong));
}

torch::Tensor SedHydroImpl::forward(torch::Tensor wr,
                                    torch::optional<torch::Tensor> out) {
  auto pcoord = phydro->pcoord;
  auto peos = phydro->peos;

  auto flux = out.value_or(torch::zeros_like(wr));

  // null-op
  if (options->sedvel()->grav()->grav1() == 0. ||
      options->sedvel()->species().size() == 0) {
    return flux;
  }

  auto vel = wr.narrow(0, IVX, 3).clone();
  coord_vec_lower_(vel, pcoord->cosine_cell_kj);

  auto temp = peos->compute("W->T", {wr});
  vsed.set_(psedvel->forward(wr[IDN], wr[IPR], temp));

  // seal top boundary
  int ie = pcoord->ie();
  int ng = vsed.size(-1) - (ie + 1);
  vsed.narrow(-1, ie + 1, ng).fill_(0.);

  // seal bottom
  int is = pcoord->is();
  vsed.slice(-1, 0, is + 1).fill_(0.);

  // 5 is number of hydro variables
  auto en = peos->compute("W->E", {wr}).index_select(0, hydro_ids - 5);

  auto rhos = wr[IDN] * wr.index_select(0, hydro_ids);
  auto rhos_vsed = rhos * vsed;

  flux.index_add_(0, hydro_ids, rhos_vsed);
  flux.narrow(0, IVX, 3) += vel * rhos_vsed.sum(0, /*keepdim=*/true);
  flux[IPR] += (vsed * en).sum(0);

  return flux;
}

std::shared_ptr<SedHydroImpl> SedHydroImpl::create(SedHydroOptions const& opts,
                                                   torch::nn::Module* p,
                                                   std::string const& name) {
  TORCH_CHECK(opts != nullptr, "SedHydro options is nullptr");
  TORCH_CHECK(p != nullptr, "Parent module is nullptr");
  return p->register_module(name, SedHydro(opts, p));
}

}  // namespace snap
