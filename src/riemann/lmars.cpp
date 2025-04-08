// base
#include <configure.h>

// snap
#include <snap/snap.h>

#include <snap/registry.hpp>

#include "riemann_formatter.hpp"
#include "riemann_solver.hpp"

namespace snap {
void call_lmars_cpu(at::TensorIterator& iter, int dim, int nvapor);
void call_lmars_cuda(at::TensorIterator& iter, int dim, int nvapor);

void LmarsSolverImpl::reset() {
  // set up equation-of-state model
  peos = register_module_op(this, "eos", options.eos());

  // set up coordinate model
  pcoord = register_module_op(this, "coord", options.coord());
}

torch::Tensor LmarsSolverImpl::forward(torch::Tensor wl, torch::Tensor wr,
                                       int dim, torch::Tensor gammad) {
  torch::NoGradGuard no_grad;

  auto out = torch::empty_like(wl);

  auto iter =
      at::TensorIteratorConfig()
          .resize_outputs(false)
          .check_all_same_dtype(true)
          .declare_static_shape(out.sizes(), /*squash_dims=*/0)
          .add_output(out)
          .add_owned_const_input(wl)
          .add_owned_const_input(wr)
          .add_owned_const_input(gammad.unsqueeze(0))
          .add_owned_const_input(peos->pthermo->cv_ratio_m1.unsqueeze(0))
          .add_owned_const_input(peos->pthermo->mu_ratio_m1.unsqueeze(0))
          .build();

  if (wl.is_cpu()) {
    call_lmars_cpu(iter, dim, peos->pthermo->options.nvapor());
  } else if (wl.is_cuda()) {
    call_lmars_cuda(iter, dim, peos->pthermo->options.nvapor());
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

  auto fepsl = peos->pthermo->f_eps(wl);
  auto fepsr = peos->pthermo->f_eps(wr);
  auto fsigl = peos->pthermo->f_sig(wl);
  auto fsigr = peos->pthermo->f_sig(wr);

  auto kappal = 1. / (gammad - 1.) * fsigl / fepsl;
  auto kappar = 1. / (gammad - 1.) * fsigr / fepsr;

  pcoord->prim2local_inplace(wl);

  auto kel = 0.5 * wl.narrow(0, IVX, 3).square().sum(0);
  auto ker = 0.5 * wr.narrow(0, IVX, 3).square().sum(0);

  // enthalpy
  auto hl = wl[IPR] / wl[IDN] * (kappal + 1.) + kel;
  auto hr = wr[IPR] / wr[IDN] * (kappar + 1.) + ker;

  auto rhobar = 0.5 * (wl[IDN] + wr[IDN]);
  auto cbar = torch::sqrt(0.5 * (1. + (1. / kappar + 1. / kappal) / 2.) *
                          (wl[IPR] + wr[IPR]) / rhobar);
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
