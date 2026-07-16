// torch
#include <ATen/Dispatch.h>
#include <ATen/TensorIterator.h>
#include <ATen/native/ReduceOpsUtils.h>
#include <ATen/native/cpu/Loops.h>

// snap
#include <snap/snap.h>

#include "riemann_dispatch.hpp"
#include "roe_impl.h"

namespace snap {

void call_roe_cpu(at::TensorIterator& iter, int dim, bool ideal_moist,
                  int nvapor, double gammad,
                  torch::Tensor const& inv_mu_ratio_m1,
                  torch::Tensor const& cv_ratio_m1, torch::Tensor const& u0) {
  int grain_size = iter.numel() / at::get_num_threads();

  AT_DISPATCH_FLOATING_TYPES(iter.dtype(), "call_roe_cpu", [&] {
    auto nhydro = at::native::ensure_nonempty_size(iter.output(0), 0);
    auto stride = at::native::ensure_nonempty_stride(iter.output(0), 0);
    auto ny = nhydro - ICY;
    auto inv_mu = ideal_moist ? inv_mu_ratio_m1.data_ptr<scalar_t>() : nullptr;
    auto cv = ideal_moist ? cv_ratio_m1.data_ptr<scalar_t>() : nullptr;
    auto energy0 = ideal_moist ? u0.data_ptr<scalar_t>() : nullptr;

    iter.for_each(
        [&](char** data, const int64_t* strides, int64_t n) {
          for (int64_t i = 0; i < n; ++i) {
            auto out = reinterpret_cast<scalar_t*>(data[0] + i * strides[0]);
            auto face_pressure =
                reinterpret_cast<scalar_t*>(data[1] + i * strides[1]);
            auto wl = reinterpret_cast<scalar_t*>(data[2] + i * strides[2]);
            auto wr = reinterpret_cast<scalar_t*>(data[3] + i * strides[3]);
            auto elr = reinterpret_cast<scalar_t*>(data[4] + i * strides[4]);
            auto glr = reinterpret_cast<scalar_t*>(data[5] + i * strides[5]);
            auto clr = reinterpret_cast<scalar_t*>(data[6] + i * strides[6]);
            roe_impl(out, wl, wr, *elr, *(elr + stride), *glr, *(glr + stride),
                     *clr, *(clr + stride), dim, ny, ideal_moist, nvapor,
                     scalar_t(gammad), inv_mu, cv, energy0, stride, stride,
                     face_pressure);
          }
        },
        grain_size);
  });
}

void call_roe_mps(at::TensorIterator& iter, int dim, bool ideal_moist,
                  int nvapor, double gammad,
                  torch::Tensor const& inv_mu_ratio_m1,
                  torch::Tensor const& cv_ratio_m1, torch::Tensor const& u0) {
  auto flx = iter.output(0);
  auto face_pressure = iter.output(1).squeeze(0);
  auto wl = iter.input(0);
  auto wr = iter.input(1);
  auto elr = iter.input(2);
  auto glr = iter.input(3);
  auto clr = iter.input(4);

  int ny = wl.size(0) - ICY;
  auto ivx = IPR - dim;
  auto ivy = IVX + ((ivx - IVX) + 1) % 3;
  auto ivz = IVX + ((ivx - IVX) + 2) % 3;
  auto scalar_view = wl.sizes().vec();
  scalar_view[0] = 1;

  auto sqrtdl = torch::sqrt(wl[IDN]);
  auto sqrtdr = torch::sqrt(wr[IDN]);
  auto isdlpdr = 1.0 / (sqrtdl + sqrtdr);
  auto rhobar = sqrtdl * sqrtdr;

  auto wroe = torch::zeros_like(wl);
  wroe[IDN] = rhobar;
  wroe.narrow(0, IVX, 3) =
      (sqrtdl * wl.narrow(0, IVX, 3) + sqrtdr * wr.narrow(0, IVX, 3)) * isdlpdr;

  auto el = elr[ILT] + 0.5 * wl[IDN] * wl.narrow(0, IVX, 3).square().sum(0);
  auto er = elr[IRT] + 0.5 * wr[IDN] * wr.narrow(0, IVX, 3).square().sum(0);
  auto hl = (el + wl[IPR]) / wl[IDN];
  auto hr = (er + wr[IPR]) / wr[IDN];
  wroe[IPR] = (hl * sqrtdl + hr * sqrtdr) * isdlpdr;

  auto ul = torch::zeros_like(wl);
  auto ur = torch::zeros_like(wr);
  auto dryl = torch::ones_like(wl[IDN]);
  auto dryr = torch::ones_like(wr[IDN]);
  if (ny > 0) {
    dryl -= wl.narrow(0, ICY, ny).sum(0);
    dryr -= wr.narrow(0, ICY, ny).sum(0);
    ul.narrow(0, ICY, ny) = wl[IDN] * wl.narrow(0, ICY, ny);
    ur.narrow(0, ICY, ny) = wr[IDN] * wr.narrow(0, ICY, ny);
  }
  ul[IDN] = wl[IDN] * dryl;
  ur[IDN] = wr[IDN] * dryr;
  ul.narrow(0, IVX, 3) = wl[IDN] * wl.narrow(0, IVX, 3);
  ur.narrow(0, IVX, 3) = wr[IDN] * wr.narrow(0, IVX, 3);
  ul[IPR] = el;
  ur[IPR] = er;

  auto fl = torch::zeros_like(wl);
  auto fr = torch::zeros_like(wr);
  fl[IDN] = ul[IDN] * wl[ivx];
  fr[IDN] = ur[IDN] * wr[ivx];
  if (ny > 0) {
    fl.narrow(0, ICY, ny) = ul.narrow(0, ICY, ny) * wl[ivx].view(scalar_view);
    fr.narrow(0, ICY, ny) = ur.narrow(0, ICY, ny) * wr[ivx].view(scalar_view);
  }
  fl.narrow(0, IVX, 3) = wl[IDN] * wl[ivx] * wl.narrow(0, IVX, 3);
  fr.narrow(0, IVX, 3) = wr[IDN] * wr[ivx] * wr.narrow(0, IVX, 3);
  fl[ivx] += wl[IPR];
  fr[ivx] += wr[IPR];
  fl[IPR] = (el + wl[IPR]) * wl[ivx];
  fr[IPR] = (er + wr[IPR]) * wr[ivx];

  auto du = ur - ul;
  auto out = 0.5 * (fl + fr);
  auto vsq = wroe.narrow(0, IVX, 3).square().sum(0);
  auto gamma_roe = 0.5 * (glr[ILT] + glr[IRT]);
  auto offset = torch::zeros_like(wroe[IPR]);
  auto qbar_dry = (ul[IDN] / sqrtdl + ur[IDN] / sqrtdr) * isdlpdr;
  auto qbar = torch::Tensor();
  auto alpha_species = torch::Tensor();
  if (ny > 0) {
    qbar = (ul.narrow(0, ICY, ny) / sqrtdl.view(scalar_view) +
            ur.narrow(0, ICY, ny) / sqrtdr.view(scalar_view)) *
           isdlpdr.view(scalar_view);
  }

  if (ideal_moist) {
    auto feps = torch::ones_like(wroe[IPR]);
    if (nvapor > 0) {
      feps += (qbar.narrow(0, 0, nvapor) *
               inv_mu_ratio_m1.narrow(0, 0, nvapor).view({nvapor, 1, 1, 1}))
                  .sum(0);
    }
    if (ny > nvapor) {
      feps -= qbar.narrow(0, nvapor, ny - nvapor).sum(0);
    }
    auto fsig = torch::ones_like(wroe[IPR]) +
                (qbar * cv_ratio_m1.view({ny, 1, 1, 1})).sum(0);
    gamma_roe = 1.0 + (gammad - 1.0) * feps / fsig;
    offset = qbar_dry * u0[0];
    if (ny > 0) {
      offset += (qbar * u0.narrow(0, 1, ny).view({ny, 1, 1, 1})).sum(0);
    }
  }

  auto q = wroe[IPR] - 0.5 * vsq - offset;
  auto cs_sq = torch::clamp_min((gamma_roe - 1.0) * q, 1.0e-10);
  auto cs = torch::sqrt(cs_sq);
  face_pressure.copy_(0.5 *
                      (wl[IPR] + wr[IPR] + rhobar * cs * (wl[ivx] - wr[ivx])));

  auto lam_m = wroe[ivx] - cs;
  auto lam_0 = wroe[ivx];
  auto lam_p = wroe[ivx] + cs;
  auto spd_m = torch::abs(lam_m);
  auto spd_0 = torch::abs(lam_0);
  auto spd_p = torch::abs(lam_p);
  auto duv = wr[ivx] - wl[ivx];
  auto dv = wr[ivy] - wl[ivy];
  auto dw = wr[ivz] - wl[ivz];
  auto dp = wr[IPR] - wl[IPR];
  auto alpha_m = -0.5 * rhobar / cs * duv + 0.5 * dp / cs_sq;
  auto alpha_p = 0.5 * rhobar / cs * duv + 0.5 * dp / cs_sq;
  auto alpha_v = rhobar * dv;
  auto alpha_w = rhobar * dw;
  auto alpha_dry = ur[IDN] - ul[IDN] - dp / cs_sq * qbar_dry;
  auto alpha0 = alpha_dry.clone();
  auto llf_flag =
      torch::logical_or(ul[IDN] + alpha_m * qbar_dry < 0.0,
                        ul[IDN] + alpha_m * qbar_dry + alpha_dry < 0.0);

  out[IDN] -= 0.5 * (spd_m * alpha_m * qbar_dry + spd_0 * alpha_dry +
                     spd_p * alpha_p * qbar_dry);
  if (ny > 0) {
    alpha_species = ur.narrow(0, ICY, ny) - ul.narrow(0, ICY, ny) -
                    dp.view(scalar_view) * qbar / cs_sq.view(scalar_view);
    alpha0 += alpha_species.sum(0);
    out.narrow(0, ICY, ny) -=
        0.5 * (spd_m.view(scalar_view) * alpha_m.view(scalar_view) * qbar +
               spd_0.view(scalar_view) * alpha_species +
               spd_p.view(scalar_view) * alpha_p.view(scalar_view) * qbar);
    auto species_first =
        ul.narrow(0, ICY, ny) + alpha_m.view(scalar_view) * qbar;
    auto species_second = species_first + alpha_species;
    llf_flag = torch::logical_or(
        llf_flag, torch::logical_or(torch::any(species_first < 0.0, 0),
                                    torch::any(species_second < 0.0, 0)));
  }

  out[ivx] -=
      0.5 * (spd_m * alpha_m * (wroe[ivx] - cs) + spd_0 * wroe[ivx] * alpha0 +
             spd_p * alpha_p * (wroe[ivx] + cs));
  out[ivy] -= 0.5 * (spd_m * alpha_m * wroe[ivy] +
                     spd_0 * (wroe[ivy] * alpha0 + alpha_v) +
                     spd_p * alpha_p * wroe[ivy]);
  out[ivz] -= 0.5 * (spd_m * alpha_m * wroe[ivz] +
                     spd_0 * (wroe[ivz] * alpha0 + alpha_w) +
                     spd_p * alpha_p * wroe[ivz]);
  out[IPR] -= 0.5 * (spd_m * alpha_m * (wroe[IPR] - wroe[ivx] * cs) +
                     spd_0 * (rhobar * (hr - hl - wroe[ivx] * duv) +
                              alpha0 * wroe[IPR] - dp) +
                     spd_p * alpha_p * (wroe[IPR] + wroe[ivx] * cs));

  auto left_upwind = (lam_m > 0).to(out.scalar_type()).view(scalar_view);
  out = left_upwind * fl + (1 - left_upwind) * out;
  auto right_upwind = (lam_p < 0).to(out.scalar_type()).view(scalar_view);
  out = right_upwind * fr + (1 - right_upwind) * out;

  auto cmax = 0.5 * torch::max(torch::abs(wl[ivx]) + clr[ILT],
                               torch::abs(wr[ivx]) + clr[IRT]);
  auto llf_mask = llf_flag.to(out.scalar_type()).view(scalar_view);
  out = llf_mask * (0.5 * (fl + fr) - cmax.view(scalar_view) * du) +
        (1 - llf_mask) * out;
  flx.copy_(out);
}

}  // namespace snap

namespace at::native {

DEFINE_DISPATCH(call_roe);
REGISTER_ALL_CPU_DISPATCH(call_roe, &snap::call_roe_cpu);
REGISTER_MPS_DISPATCH(call_roe, &snap::call_roe_mps);

}  // namespace at::native
