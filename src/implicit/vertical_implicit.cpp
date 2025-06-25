// yaml
#include <yaml-cpp/yaml.h>

// torch
#include <ATen/TensorIterator.h>

// snap
#include <snap/snap.h>

#include <snap/registry.hpp>

#include "vertical_implicit.hpp"

namespace snap {
template <int N>
__attribute__((weak)) void vic_forward_cuda(at::TensorIterator& iter, double dt,
                                            int il, int iu) {}

template <int N>
void vic_forward_cpu(at::TensorIterator& iter, double dt, int il, int iu) {}

ImplicitOptions ImplicitOptions::from_yaml(const YAML::Node& root) {
  ImplicitOptions op;

  if (!root["dynamics"]) return op;
  if (!root["dynamics"]["integrator"]) return op;

  switch (root["dynamics"]["integrator"]["implicit-scheme"].as<int>(0)) {
    case 0:
      op.type() = "none";
      op.scheme() = 0;
      break;
    case 1:
      op.type() = "vic-partial";
      op.scheme() = 1;
      break;
    case 9:
      op.type() = "vic-full";
      op.scheme() = 9;
      break;
    default:
      TORCH_CHECK(false, "Unsupported implicit scheme");
  }

  if (!root["geometry"]) return op;
  if (!root["geometry"]["cells"]) return op;
  op.nghost() = root["geometry"]["cells"]["nghost"].as<int>(1);

  if (!root["forcing"]) return op;
  if (!root["forcing"]["const-gravity"]) return op;

  op.grav() = root["forcing"]["const-gravity"]["grav1"].as<double>(0.0);

  return op;
}

VerticalImplicitImpl::VerticalImplicitImpl(ImplicitOptions options_)
    : options(options_) {
  reset();
}

void VerticalImplicitImpl::reset() {
  // set up reconstruct model
  auto op = ReconstructOptions();
  precon = register_module("recon", Reconstruct(options.recon()));

  // set up coordinate model
  pcoord = register_module_op(this, "coord", options.coord());
}

torch::Tensor sound_speed_ideal_gas(torch::Tensor w, torch::Tensor gm1) {
  return torch::sqrt((1. + gm1) * w[Index::IPR] / w[Index::IDN]);
}

torch::Tensor VerticalImplicitImpl::diffusion_matrix(torch::Tensor w,
                                                     torch::Tensor gm1) {
  enum { DIM1 = 3 };
  auto wlr = precon->forward(w, DIM1);
  auto wroe = roe_average(wlr, gm1);
  auto cs = sound_speed_ideal_gas(wroe, gm1);
  auto [Rmat, Rimat] = eigen_vectors(wroe, gm1, cs);
  auto vel = wroe[Index::IVX];
  auto nc1 = w.size(3);
  auto b = Rmat * torch::stack({vel - cs, vel, vel + cs, vel, vel}).abs();
  return torch::einsum("jk...,kl...->...jl",
                       {b.view({5, 5, -1, nc1}), Rimat.view({5, 5, -1, nc1})});
}

torch::Tensor VerticalImplicitImpl::forward(torch::Tensor w, torch::Tensor du,
                                            torch::Tensor gm1, double dt) {
  torch::NoGradGuard no_grad;

  if (options.scheme() == 0) {  // null operation
    return du;
  }

  int msize;
  if (options.scheme() == 1) {  // partial
    msize = 3;
  } else if (options.scheme() == 9) {  // full
    msize = 5;
  } else {
    throw std::runtime_error("Unsupported scheme");
  }

  auto Dt = torch::eye(msize, w.options()).view({1, 1, msize, msize}) * 1. / dt;

  auto Phi = torch::zeros({msize, msize}, w.options());
  Phi[Index::IVX][Index::IDN] = options.grav();
  Phi[msize - 1][Index::IVX] = options.grav();
  Phi = Phi.view({1, 1, msize, msize});

  auto Bnd = torch::eye(msize, w.options());
  Bnd[Index::IVX][Index::IVX] = -1.;

  //// -------------- Populating Matrix -------------- ////
  int is = options.nghost();
  int ie = w.size(3) - options.nghost();

  auto area = pcoord->face_area1();
  auto aleft = area.slice(2, is, ie).view({-1, ie - is, 1, 1});
  auto aright = area.slice(2, is + 1, ie + 1).view({-1, ie - is, 1, 1});
  auto vol = pcoord->cell_volume().slice(2, is, ie).view({-1, ie - is, 1, 1});

  torch::Tensor A, dfdq;

  // Indices for a 3x3 submatrix
  if (options.scheme() == 1) {  // partial matrix
    auto sub =
        torch::tensor({0, 1, 4}, torch::dtype(torch::kLong).device(w.device()));
    A = diffusion_matrix(w, gm1).index_select(2, sub).index_select(3, sub);
    dfdq = flux_jacobian(w, gm1)
               .view({5, 5, -1, w.size(3)})
               .index_select(0, sub)
               .index_select(1, sub)
               .permute({2, 3, 0, 1});
  } else if (options.scheme() == 9) {  // full matrix
    A = diffusion_matrix(w, gm1);
    dfdq =
        flux_jacobian(w, gm1).view({5, 5, -1, w.size(3)}).permute({2, 3, 0, 1});
  } else {
    throw std::runtime_error("Unsupported scheme");
  }

  //// ------------ Allocate solver memory ------------ ////
  int nhydro = w.size(0);
  int ncol = w.size(1) * w.size(2);
  int nlayer = w.size(3);
  auto a = torch::zeros({ncol, nlayer, msize, msize}, w.options());
  auto b = torch::zeros({ncol, nlayer, msize, msize}, w.options());
  auto c = torch::zeros({ncol, nlayer, msize, msize}, w.options());
  auto delta = torch::zeros({ncol, nlayer, msize}, w.options());
  auto corr = torch::zeros({ncol, nlayer, msize}, w.options());

  a.slice(1, is, ie) =
      (A.slice(1, is, ie) * aleft + A.slice(1, is + 1, ie + 1) * aright +
       (aright - aleft) * dfdq.slice(1, is, ie)) /
          (2. * vol) +
      Dt - Phi;

  b.slice(1, is, ie) =
      -(A.slice(1, is - 1, ie - 1) + dfdq.slice(1, is - 1, ie - 1)) * aleft /
      (2. * vol);

  c.slice(1, is, ie) =
      -(A.slice(1, is + 1, ie + 1) - dfdq.slice(1, is + 1, ie + 1)) * aright /
      (2. * vol);

  corr.slice(1, is, ie) = 0.;

  //// ----------- Fix boundary condition ------------ ////
  a.select(1, is) += torch::matmul(b.select(1, is), Bnd);
  a.select(1, ie - 1) += torch::matmul(c.select(1, ie - 1), Bnd);

  //// -------- Solve block-tridiagonal matrix--------- ////
  auto iter =
      at::TensorIteratorConfig()
          .resize_outputs(false)
          .check_all_same_dtype(true)
          .declare_static_shape({nhydro, ncol, nlayer}, /*squash_dims=*/{0, 2})
          .add_owned_output(du.view({nhydro, -1, nlayer}))
          .add_owned_input(w.view({nhydro, -1, nlayer}))
          .add_owned_input(a.view({ncol, nlayer, -1}).permute({2, 0, 1}))
          .add_owned_input(b.view({ncol, nlayer, -1}).permute({2, 0, 1}))
          .add_owned_input(c.view({ncol, nlayer, -1}).permute({2, 0, 1}))
          .add_owned_input(delta.permute({2, 0, 1}))
          .add_owned_input(corr.permute({2, 0, 1}))
          .build();

  if (w.is_cpu()) {
    if (msize == 3) {
      vic_forward_cpu<3>(iter, dt, is, ie - 1);
    } else {
      vic_forward_cpu<5>(iter, dt, is, ie - 1);
    }
  } else if (w.is_cuda()) {
    if (msize == 3) {
      vic_forward_cuda<3>(iter, dt, is, ie - 1);
    } else {
      vic_forward_cuda<5>(iter, dt, is, ie - 1);
    }
  } else {
    TORCH_CHECK(false, "Unsupported device type");
  }

  return du;
}

torch::Tensor roe_average(torch::Tensor wlr, torch::Tensor gm1) {
  using Index::IDN;
  using Index::ILT;
  using Index::IPR;
  using Index::IRT;
  using Index::IVX;

  auto sqrtdlr = torch::sqrt(wlr.select(1, IDN));
  auto isdlpdr = 1.0 / (sqrtdlr[ILT] + sqrtdlr[IRT]);

  auto roe = torch::zeros_like(wlr[0]);

  roe[IDN] = sqrtdlr[ILT] * sqrtdlr[IRT];

  roe.narrow(0, IVX, 3) =
      (sqrtdlr.unsqueeze(1) * wlr.narrow(1, IVX, 3)).sum(0) * isdlpdr;

  auto kelr = 0.5 * wlr.narrow(1, IVX, 3).square().sum(1);

  // Etot of the left/right side.
  auto elr = wlr.select(1, IPR) / gm1 + kelr * wlr.select(1, IDN);

  // Enthalpy divided by the density.
  auto hbar = ((elr + wlr.select(1, IPR)) / sqrtdlr).sum(0) * isdlpdr;

  // Roe averaged pressure
  auto ke = 0.5 * roe.narrow(0, IVX, 3).square().sum(0);

  roe[IPR] = (hbar - ke) * gm1 / (gm1 + 1.) * roe[IDN];

  return roe;
}

torch::Tensor flux_jacobian(torch::Tensor w, torch::Tensor gm1) {
  using Index::IDN;
  using Index::IPR;
  using Index::IVX;
  using Index::IVY;
  using Index::IVZ;

  auto v1 = w[IVX];
  auto v2 = w[IVY];
  auto v3 = w[IVZ];
  auto rho = w[IDN];
  auto pres = w[IPR];

  auto s2 = w.narrow(0, IVX, 3).square().sum(0);
  auto c1 = ((gm1 - 1) * s2 / 2. - (gm1 + 1) / gm1 * pres / rho) * v1;
  auto c2 = (gm1 + 1) / gm1 * pres / rho + s2 / 2. - gm1 * v1 * v1;

  auto zeros = torch::zeros_like(w[0]);
  auto ones = torch::ones_like(w[0]);

  auto result = torch::stack(
      {torch::stack({zeros, ones, zeros, zeros, zeros}),
       torch::stack({gm1 * s2 / 2 - v1 * v1, (2. - gm1) * v1, -gm1 * v2,
                     -gm1 * v3, gm1}),
       torch::stack({-v1 * v2, v2, v1, zeros, zeros}),
       torch::stack({-v1 * v3, v3, zeros, v1, zeros}),
       torch::stack({c1, c2, -gm1 * v2 * v1, -gm1 * v3 * v1, (gm1 + 1) * v1})});
  return result;
}

std::pair<torch::Tensor, torch::Tensor> eigen_vectors(torch::Tensor prim,
                                                      torch::Tensor gm1,
                                                      torch::Tensor cs) {
  using Index::IDN;
  using Index::IPR;
  using Index::IVX;
  using Index::IVY;
  using Index::IVZ;

  auto r = prim[IDN];
  auto u = prim[IVX];
  auto v = prim[IVY];
  auto w = prim[IVZ];
  auto p = prim[IPR];

  auto ke = 0.5 * prim.narrow(0, IVX, 3).square().sum(0);
  auto hp = (gm1 + 1.) / gm1 * p / r;
  auto h = hp + ke;

  auto zeros = torch::zeros_like(w);
  auto ones = torch::ones_like(w);

  auto result1 =
      torch::stack({torch::stack({ones, ones, ones, zeros, zeros}),
                    torch::stack({u - cs, u, u + cs, zeros, zeros}),
                    torch::stack({v, v, v, ones, zeros}),
                    torch::stack({w, w, w, zeros, ones}),
                    torch::stack({h - u * cs, ke, h + u * cs, v, w})});

  auto result2 = torch::stack(
      {torch::stack({(cs * ke + u * hp) / (2. * cs * hp),
                     (-hp - cs * u) / (2. * cs * hp), -v / (2. * hp),
                     -w / (2. * hp), 1. / (2. * hp)}),
       torch::stack({(hp - ke) / hp, u / hp, v / hp, w / hp, -1. / hp}),
       torch::stack({(cs * ke - u * hp) / (2. * cs * hp),
                     (hp - cs * u) / (2. * cs * hp), -v / (2. * hp),
                     -w / (2. * hp), 1. / (2. * hp)}),
       torch::stack({-v, zeros, ones, zeros, zeros}),
       torch::stack({-w, zeros, zeros, ones, zeros})});

  return std::make_pair(result1, result2);
}
}  // namespace snap
