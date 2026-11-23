#include "sphere_transport.hpp"

#include <snap/bc/bc_func.hpp>
#include <snap/coord/coord_utils.hpp>
#include <snap/coord/cubed_sphere_utils.hpp>
#include <snap/eb/geometry.hpp>
#include <snap/mesh/meshblock.hpp>

namespace snap {
namespace {
using Tensor = torch::Tensor;
struct Basis {
  Tensor alpha, beta, cosine;
  int panel;
  Basis(const MeshBlockImpl& b, const Tensor& vector, int d) {
    auto c = b.pcoord;
    alpha = (d == 2 ? c->x2f : c->x2v)
                .narrow(0, 0, vector.size(-2))
                .view({1, -1, 1})
                .to(vector.options());
    beta = (d == 3 ? c->x3f : c->x3v)
               .narrow(0, 0, vector.size(-3))
               .view({-1, 1, 1})
               .to(vector.options());
    cosine = (d == 2   ? c->cosine_face2_kj
              : d == 3 ? c->cosine_face3_kj
                       : c->cosine_cell_kj)
                 .narrow(0, 0, vector.size(-3))
                 .narrow(1, 0, vector.size(-2))
                 .to(vector.options());
    panel = b.eb_geometry()->panel;
  }
  Tensor to_xyz(Tensor v, bool covariant) const {
    auto out = v.clone();
    if (covariant) coord_vec_raise_(out, cosine);
    cs_contra_to_cart_(out, alpha, beta, panel);
    // Snapy's Cartesian velocity storage orders the vertical axis first
    // (Z,X,Y). Geometry stores conventional global coordinates (X,Y,Z).
    return torch::stack({out[1], out[2], out[0]});
  }
  Tensor from_xyz(const Tensor& v, bool covariant) const {
    auto out = torch::stack({v[2], v[0], v[1]});
    cs_cart_to_contra_(out, alpha, beta, panel);
    if (covariant) coord_vec_lower_(out, cosine);
    return out;
  }
};
}  // namespace

Tensor sphere_momentum_to_cart(const MeshBlockImpl& b, const Tensor& v, int d) {
  return Basis(b, v, d).to_xyz(v, true);
}
Tensor sphere_cart_to_momentum(const MeshBlockImpl& b, const Tensor& v, int d) {
  return Basis(b, v, d).from_xyz(v, true);
}
Tensor sphere_velocity_to_cart(const MeshBlockImpl& b, const Tensor& v, int d) {
  return Basis(b, v, d).to_xyz(v, false);
}
Tensor sphere_cart_to_velocity(const MeshBlockImpl& b, const Tensor& v, int d) {
  return Basis(b, v, d).from_xyz(v, false);
}

void sphere_riemann(HydroImpl& hydro, const Tensor& lr, int dim, Tensor flux,
                    Tensor pressure, Tensor central) {
  const int d = 4 - dim;
  const auto& eb = *hydro.pmb->eb_geometry();
  auto av = eb.face_area_vector[d - 1]
                .narrow(dim, 0, flux.size(dim))
                .to(flux.options());
  auto area = av.norm(2, 0), denom = torch::where(area > 0, area, 1.);
  auto n = av / denom;
  // The frame at a closed face is arbitrary; its flux has zero measure.
  n[0] = torch::where(area > 0, n[0], 1.);
  auto unit = torch::zeros_like(n);
  auto use_x = n[0].abs() < 0.8;
  unit[0] = use_x.to(n.options());
  unit[1] = (~use_x).to(n.options());
  auto t = torch::cross(n, unit, 0);
  t = t / t.norm(2, 0);
  auto s = torch::cross(n, t, 0);
  Basis basis(*hydro.pmb, n, d);
  auto oriented = lr.clone();
  int normal = d - 1, first = (normal + 1) % 3, second = (normal + 2) % 3;
  for (int side : {ILT, IRT}) {
    auto v = oriented[side].narrow(0, IVX, 3);
    auto xyz = basis.to_xyz(v, false);
    v[normal].copy_((xyz * n).sum(0));
    v[first].copy_((xyz * t).sum(0));
    v[second].copy_((xyz * s).sum(0));
  }
  if (d == 1) {
    auto c = hydro.pmb->pcoord;
    for (int end : {0, 1}) {
      auto const& bc = hydro.pmb->options->bfuncs();
      if (bc.size() < 2 || !bc[end] || !is_reflecting(bc[end])) continue;
      auto face = end ? c->iu() + 1 : c->il();
      auto inside = oriented[end ? ILT : IRT].select(-1, face);
      auto outside = oriented[end ? IRT : ILT].select(-1, face);
      outside.copy_(inside);
      outside[IVX].neg_();
    }
  }
  hydro.priemann->orthonormal(oriented[ILT], oriented[IRT], dim, flux, pressure,
                              central);
  auto m = flux.narrow(0, IVX, 3);
  m.copy_(basis.from_xyz(m[normal] * n + m[first] * t + m[second] * s, true));
  auto aperture = eb.face_fraction[d - 1].narrow(dim - 1, 0, flux.size(dim));
  auto open_area =
      (eb.background_face_area[d - 1].narrow(dim - 1, 0, flux.size(dim)) *
       aperture)
          .to(area);
  auto scale = area / torch::where(open_area > 0, open_area, 1.);
  flux.mul_(scale);
  if (central.defined()) central.mul_(scale);
}

Tensor sphere_radial_wall_traction(HydroImpl& hydro, const Tensor& w) {
  // A single-layer shell has two impermeable radial faces even though
  // its radial sweep is inactive. Each face sees its own outward velocity.
  const auto& b = *hydro.pmb;
  auto radial = b.eb_geometry()->face_area_vector[0].to(w.options());
  auto velocity = sphere_velocity_to_cart(b, w.narrow(0, IVX, 3));
  auto traction = torch::zeros_like(velocity);
  for (int side : {0, 1}) {
    auto area = (side ? 1. : -1.) * radial.narrow(-1, side, 1);
    auto len = area.norm(2, 0);
    auto wall = w.clone();
    wall.narrow(0, IVX, 3).zero_();
    wall[IVX].copy_((velocity * area).sum(0) / torch::where(len > 0, len, 1.));
    auto mirror = wall.clone();
    mirror[IVX].neg_();
    auto flux = torch::zeros_like(wall);
    hydro.priemann->orthonormal(wall, mirror, 3, flux);
    traction.add_(flux[IVX] * area);
  }
  return traction;
}

Tensor sphere_flux_divergence(HydroImpl& hydro, const Tensor& w,
                              std::array<Tensor, 3> fluxes) {
  auto& block = *hydro.pmb;
  auto c = block.pcoord;
  for (int d = 1; d <= 3; ++d)
    if (fluxes[d - 1].defined()) {
      auto f = fluxes[d - 1].clone();
      f.narrow(0, IVX, 3).copy_(
          sphere_momentum_to_cart(block, f.narrow(0, IVX, 3), d));
      fluxes[d - 1] = f;
    }
  auto div = c->CoordinateImpl::forward(w, fluxes[0], fluxes[1], fluxes[2]);
  auto m = div.narrow(0, IVX, 3);
  m.copy_(sphere_cart_to_momentum(block, m));
  return div;
}
}  // namespace snap
