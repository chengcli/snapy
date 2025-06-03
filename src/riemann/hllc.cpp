// base
#include <configure.h>

// snap
#include <snap/snap.h>

#include <snap/registry.hpp>

#include "riemann_formatter.hpp"
#include "riemann_solver.hpp"

namespace snap {
void call_hllc_cpu(at::TensorIterator& iter, int dim, int nvapor);
__attribute__((weak)) void call_hllc_cuda(at::TensorIterator& iter, int dim,
                                          int nvapor) {}

void HLLCSolverImpl::reset() {
  // set up equation-of-state model
  peos = register_module_op(this, "eos", options.eos());

  // set up coordinate model
  pcoord = register_module_op(this, "coord", options.coord());
}

torch::Tensor HLLCSolverImpl::forward(torch::Tensor wl, torch::Tensor wr,
                                      int dim, torch::Tensor gammad) {
  torch::NoGradGuard no_grad;

  auto out = torch::empty_like(wl);

  auto iter =
      at::TensorIteratorConfig()
          .resize_outputs(false)
          .check_all_same_dtype(false)
          .declare_static_shape(out.sizes(), /*squash_dims=*/0)
          .add_output(out)
          .add_owned_const_input(wl)
          .add_owned_const_input(wr)
          .add_owned_const_input(gammad.unsqueeze(0))
          .add_owned_const_input(peos->pthermo->cv_ratio_m1.unsqueeze(0))
          .add_owned_const_input(peos->pthermo->mu_ratio_m1.unsqueeze(0))
          .build();

  if (wl.is_cpu()) {
    call_hllc_cpu(iter, dim, peos->pthermo->options.vapor_ids().size());
  } else if (wl.is_cuda()) {
    call_hllc_cuda(iter, dim, peos->pthermo->options.vapor_ids().size());
  } else {
    return forward_fallback(wl, wr, dim, gammad);
  }

  return out;
}

torch::Tensor HLLCSolverImpl::forward_fallback(torch::Tensor wl,
                                               torch::Tensor wr, int dim,
                                               torch::Tensor gammad) {
  using Index::IDN;
  using Index::IPR;
  using Index::IVX;

  auto TINY_NUMBER = 1.0e-10;

  // dim, ivx, ivy, ivz
  // 3, IVX, IVY, iVZ
  // 2, IVX + 1, IVX + 2, IVX
  // 1, IVX + 2, IVX, IVX + 1
  auto ivx = IPR - dim;
  auto ivy = IVX + ((ivx - IVX) + 1) % 3;
  auto ivz = IVX + ((ivx - IVX) + 2) % 3;

  auto igm1 = 1.0 / (gammad - 1.0);

  //--- Step 2.  Compute middle state estimates with PVRS (Toro 10.5.2)

  auto cl = peos->sound_speed(wl);
  auto cr = peos->sound_speed(wr);

  auto el =
      wl[IPR] * igm1 + 0.5 * wl[IDN] * wl.narrow(0, IVX, 3).square().sum(0);
  auto er =
      wr[IPR] * igm1 + 0.5 * wr[IDN] * wr.narrow(0, IVX, 3).square().sum(0);

  auto rhoa = .5 * (wl[IDN] + wr[IDN]);  // average density
  auto ca = .5 * (cl + cr);              // average sound speed
  auto pmid = .5 * (wl[IPR] + wr[IPR] + (wl[ivx] - wr[ivx]) * rhoa * ca);
  auto umid = .5 * (wl[ivx] + wr[ivx] + (wl[IPR] - wr[IPR]) / (rhoa * ca));

  //--- Step 3.  Compute sound speed in L,R

  auto ql =
      torch::sqrt(1.0 + (gammad + 1) / (2 * gammad) * (pmid / wl[IPR] - 1.0));
  ql = torch::where(pmid <= wl[IPR], 1., ql);

  auto qr =
      torch::sqrt(1.0 + (gammad + 1) / (2 * gammad) * (pmid / wr[IPR] - 1.0));
  qr = torch::where(pmid <= wr[IPR], 1., qr);

  //--- Step 4.  Compute the max/min wave speeds based on L/R

  auto al = wl[ivx] - cl * ql;
  auto ar = wr[ivx] + cr * qr;

  auto bp = torch::where(ar > 0.0, ar, TINY_NUMBER);
  auto bm = torch::where(al < 0.0, al, -TINY_NUMBER);

  //--- Step 5. Compute the contact wave speed and pressure

  auto vxl = wl[ivx] - al;
  auto vxr = wr[ivx] - ar;

  auto tl = wl[IPR] + vxl * wl[IDN] * wl[ivx];
  auto tr = wr[IPR] + vxr * wr[IDN] * wr[ivx];

  auto ml = wl[IDN] * vxl;
  auto mr = -(wr[IDN] * vxr);

  // Determine the contact wave speed...
  auto am = (tl - tr) / (ml + mr);
  // ...and the pressure at the contact surface
  auto cp = (ml * tr + mr * tl) / (ml + mr);
  cp = torch::where(cp > 0.0, cp, 0.0);

  //--- Step 6. Compute L/R fluxes along the line bm, bp
  auto fl = torch::zeros_like(wl);
  auto fr = torch::zeros_like(wr);

  vxl = wl[ivx] - bm;
  vxr = wr[ivx] - bp;

  fl[IDN] = wl[IDN] * vxl;
  fr[IDN] = wr[IDN] * vxr;

  fl.narrow(0, IVX, 3) = wl[IDN] * wl.narrow(0, IVX, 3) * vxl;
  fr.narrow(0, IVX, 3) = wr[IDN] * wr.narrow(0, IVX, 3) * vxr;

  fl[ivx] += wl[IPR];
  fr[ivx] += wr[IPR];

  fl[IPR] = el * vxl + wl[IPR] * wl[ivx];
  fr[IPR] = er * vxr + wr[IPR] * wr[ivx];

  //--- Step 8. Compute flux weights or scales

  auto ami = am >= 0.0;
  auto sl = torch::where(ami, am / (am - bm), 0.0);
  auto sr = torch::where(ami, 0.0, -am / (bp - am));
  auto sm = torch::where(ami, -bm / (am - bm), bp / (bp - am));

  //--- Step 9. Compute the HLLC flux at interface, including weighted
  // contribution
  // of the flux along the contact

  auto flx = sl.unsqueeze(0) * fl + sr.unsqueeze(0) * fr;
  flx[ivx] += sm * cp;
  flx[IPR] += sm * cp * am;

  return flx;
}
}  // namespace snap
