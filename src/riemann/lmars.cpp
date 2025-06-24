// snap
#include <snap/snap.h>

#include <snap/registry.hpp>

#include "riemann_solver.hpp"

namespace snap {
void call_lmars_cpu(at::TensorIterator& iter, int dim, int nvapor);
__attribute__((weak)) void call_lmars_cuda(at::TensorIterator& iter, int dim,
                                           int nvapor) {}

void LmarsSolverImpl::reset() {
  // set up equation-of-state model
  peos = register_module_op(this, "eos", options.eos());
}

torch::Tensor LmarsSolverImpl::forward(torch::Tensor wl, torch::Tensor wr,
                                       int dim, torch::Tensor gammad) {
  auto out = torch::empty_like(wl);

  // FIXME(cli): This is a place holder
  auto cv_ratio_m1 = torch::ones(peos->nhydro() - 5, wl.options());
  auto mu_ratio_m1 = torch::ones(peos->nhydro() - 5, wl.options());

  auto iter = at::TensorIteratorConfig()
                  .resize_outputs(false)
                  .check_all_same_dtype(true)
                  .declare_static_shape(out.sizes(), /*squash_dims=*/0)
                  .add_output(out)
                  .add_owned_const_input(wl)
                  .add_owned_const_input(wr)
                  .add_owned_const_input(gammad.unsqueeze(0))
                  .add_owned_const_input(cv_ratio_m1.unsqueeze(0))
                  .add_owned_const_input(mu_ratio_m1.unsqueeze(0))
                  .build();

  if (wl.is_cpu()) {
    call_lmars_cpu(iter, dim, peos->options.thermo().vapor_ids().size());
  } else if (wl.is_cuda()) {
    call_lmars_cuda(iter, dim, peos->options.thermo().vapor_ids().size());
  } else {
    return forward_fallback(wl, wr, dim, gammad);
  }

  return out;
}

torch::Tensor LmarsSolverImpl::forward_fallback(torch::Tensor wl,
                                                torch::Tensor wr, int dim,
                                                torch::Tensor gammad) {
  using Index::ICY;
  using Index::IDN;
  using Index::IPR;
  using Index::IVX;
  int ny = wl.size(0) - 5;

  // dim, ivx
  // 3, IVX
  // 2, IVX + 1
  // 1, IVX + 2
  auto ivx = IPR - dim;
  auto ivy = IVX + ((ivx - IVX) + 1) % 3;
  auto ivz = IVX + ((ivx - IVX) + 2) % 3;

  auto el = peos->compute("W->U", {wl});
  auto gammal = peos->compute("W->A", {wl});

  auto er = peos->compute("W->U", {wr});
  auto gammar = peos->compute("W->A", {wr});

  pcoord->prim2local_inplace(wl);

  auto kel = 0.5 * wl.narrow(0, IVX, 3).square().sum(0);
  auto ker = 0.5 * wr.narrow(0, IVX, 3).square().sum(0);

  // enthalpy
  auto hl = el + wl[IPR];
  auto hr = er + wr[IPR];

  auto rhobar = 0.5 * (wl[IDN] + wr[IDN]);
  auto gamma_bar = 0.5 * (gammal + gammar);
  auto cbar = torch::sqrt(gamma_bar * 0.5 * (wl[IPR] + wr[IPR]) / rhobar);
  auto pbar =
      0.5 * (wl[IPR] + wr[IPR]) + 0.5 * (rhobar * cbar) * (wl[ivx] - wr[ivx]);
  auto ubar =
      0.5 * (wl[ivx] + wr[ivx]) + 0.5 / (rhobar * cbar) * (wl[IPR] - wr[IPR]);

  // left flux
  auto fluxl = torch::zeros_like(wl);
  auto fluxr = torch::zeros_like(wr);

  fluxl[IDN] = ubar * wl[IDN] *
               (torch::ones_like(wl[IDN]) - wl.narrow(0, ICY, ny).sum(0));
  fluxl.narrow(0, ICY, ny) = ubar * wl[IDN] * wl.narrow(0, ICY, ny);
  fluxl.narrow(0, IVX, 3) = ubar * wl[IDN] * wl.narrow(0, IVX, 3);
  fluxl[ivx] += pbar;
  fluxl[IPR] = ubar * wl[IDN] * hl;

  // right flux
  fluxr[IDN] = ubar * wr[IDN] *
               (torch::ones_like(wr[IDN]) - wr.narrow(0, ICY, ny).sum(0));
  fluxr.narrow(0, ICY, ny) = ubar * wr[IDN] * wr.narrow(0, ICY, ny);
  fluxr.narrow(0, IVX, 3) = ubar * wr[IDN] * wr.narrow(0, IVX, 3);
  fluxr[ivx] += pbar;
  fluxr[IPR] = ubar * wr[IDN] * hr;

  auto ui = (ubar > 0).to(torch::kInt);
  auto flx = ui * fluxl + (1 - ui) * fluxr;
  pcoord->flux2global_inplace(flx);

  return flx;
}
}  // namespace snap
