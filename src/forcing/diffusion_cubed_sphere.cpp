// Conservative physical diffusion on a nonorthogonal cubed sphere.
#include <array>
#include <cmath>
#include <snap/coord/coord_utils.hpp>
#include <snap/coord/cubed_sphere_utils.hpp>
#include <snap/mesh/meshblock.hpp>

#include "forcing.hpp"

namespace snap {
namespace {
using Box = std::array<int64_t, 3>;  // k,j,i
using Tensor = torch::Tensor;
Tensor sub(Tensor t, Box lo, Box hi) {
  for (int d = 0; d < 3; ++d) t = t.narrow(d - 3, lo[d], hi[d] - lo[d]);
  return t;
}
Tensor vector_dot(Tensor a, Tensor b) { return (a * b).sum(0); }
std::array<Tensor, 3> basis(Tensor a, Tensor b, Tensor r, int panel) {
  auto shape = torch::broadcast_tensors({a, b, r})[0];
  std::array<Tensor, 3> out;
  for (int d = 0; d < 3; ++d) {
    out[d] = torch::zeros({3, shape.size(0), shape.size(1), shape.size(2)},
                          shape.options());
    out[d][d].fill_(1.);
    cs_contra_to_cart_(out[d], a, b, panel);
  }
  return out;
}

// Four-point polynomial weights. At block boundaries use inward stencils,
// never an unfilled corner or an interpolated neighbor-panel coordinate.
// Cubic traces make the seam jump penalty O(h^3) on a smooth solution.
Tensor weights(int n, bool faces, bool derivative, double h, Tensor like) {
  auto w = torch::zeros({n + (faces ? 1 : 0), n}, torch::kFloat64);
  auto ptr = w.accessor<double, 2>();
  for (int row = 0; row < w.size(0); ++row) {
    double q = row - (faces ? 0.5 : 0.);
    int start =
        std::max(0, std::min(n - 4, static_cast<int>(std::floor(q)) - 1));
    for (int j = start; j < start + 4; ++j) {
      double value = 0.;
      if (!derivative) {
        value = 1.;
        for (int k = start; k < start + 4; ++k)
          if (k != j) value *= (q - k) / (j - k);
      } else
        for (int k = start; k < start + 4; ++k)
          if (k != j) {
            double term = 1. / (j - k);
            for (int l = start; l < start + 4; ++l)
              if (l != j && l != k) term *= (q - l) / (j - l);
            value += term / h;
          }
      ptr[row][j] = value;
    }
  }
  return w.to(like.options());
}
Tensor transform(Tensor t, Tensor matrix, int dim) {
  return torch::matmul(t.movedim(dim, -1), matrix.transpose(0, 1))
      .movedim(-1, dim);
}
constexpr std::array<std::array<int, 3>, 4> offsets = {
    {{0, -1, 0}, {0, 1, 0}, {-1, 0, 0}, {1, 0, 0}}};

// Exchange outward integrated flux, reconstructed trace, dynamic coefficients,
// and centre-to-face normal distance. Cartesian channels require no rotation.
// Opposite signs of a common flux guarantee conservation at every block seam.
void share_fluxes(MeshBlockImpl const *block, std::array<Tensor, 3> &flux,
                  std::array<Tensor, 3> const &trace,
                  std::array<Tensor, 3> const &normals) {
  auto c = block->pcoord;
  Box lo{c->kl(), c->jl(), c->il()}, hi{c->ku() + 1, c->ju() + 1, c->iu() + 1};
  auto packed =
      torch::zeros({48, flux[0].size(1), flux[0].size(2), flux[0].size(3)},
                   flux[0].options());
  std::array<Tensor, 3> area{c->face_area1(), c->face_area2(), c->face_area3()};
  for (int side = 0; side < 4; ++side) {
    int dir = side < 2 ? 1 : 2, dim = 2 - dir;
    bool upper = side % 2;
    Box cl = lo, ch = hi, fl = lo, fh = hi;
    cl[dim] = upper ? hi[dim] - 1 : lo[dim];
    ch[dim] = cl[dim] + 1;
    fl[dim] = upper ? hi[dim] : lo[dim];
    fh[dim] = fl[dim] + 1;
    auto p = sub(packed.narrow(0, 12 * side, 12), cl, ch);
    p.narrow(0, 0, 5).copy_((upper ? 1. : -1.) * sub(flux[dir], fl, fh) *
                            sub(area[dir], fl, fh));
    p.narrow(0, 5, 7).copy_(sub(trace[dir], fl, fh));
  }
  Variables vars{{"diffusion_flux", packed}};
  block->exchange(vars, SyncOptions().type(kScalar).phyid(12));
  auto layout = block->get_layout();
  auto loc = layout->loc_of(layout->options->rank());
  for (int side = 0; side < 4; ++side) {
    auto off = std::make_tuple(offsets[side][0], offsets[side][1], 0);
    auto peer = layout->loc_of(layout->neighbor_rank(loc, off));
    int ps = std::get<2>(peer) == std::get<2>(loc)
                 ? (side ^ 1)
                 : CS_FACE_EDGES[std::get<2>(loc)][side].nside;
    int dir = side < 2 ? 1 : 2, dim = 2 - dir;
    bool upper = side % 2;
    double sign = upper ? 1. : -1.;
    Box gl = lo, gh = hi, fl = lo, fh = hi;
    gl[dim] = upper ? hi[dim] : lo[dim] - 1;
    gh[dim] = gl[dim] + 1;
    fl[dim] = upper ? hi[dim] : lo[dim];
    fh[dim] = fl[dim] + 1;
    auto other = sub(packed.narrow(0, 12 * ps, 12), gl, gh);
    auto own = sub(trace[dir], fl, fh);
    auto f = sub(flux[dir], fl, fh);
    f.copy_(0.5 * (f - sign * other.narrow(0, 0, 5) / sub(area[dir], fl, fh)));
    auto n = sign * sub(normals[dir], fl, fh);
    auto jump = other.narrow(0, 5, 4) - own.narrow(0, 0, 4);
    auto distance = own[6] + other[11];
    auto mu = 0.5 * (own[4] + other[9]), k = 0.5 * (own[5] + other[10]);
    // Symmetric interior penalty couples even piecewise-constant panels.
    auto dv = jump.narrow(0, 1, 3);
    auto mf = -2. * mu * (dv + n * vector_dot(n, dv) / 3.) / distance;
    auto vf = 0.5 * (own.narrow(0, 1, 3) + other.narrow(0, 6, 3));
    f.narrow(0, IVX, 3).add_(sign * mf);
    f[IPR].add_(sign * (-2. * k * jump[0] / distance + vector_dot(vf, mf)));
  }
}
}  // namespace

Tensor DiffusionImpl::forward_cubed_sphere(Tensor du, Tensor w, Tensor temp,
                                           double dt) {
  auto block = phydro->pmb;
  auto c = block->pcoord;
  auto op = c->options;
  int panel = std::get<2>(
      block->get_layout()->loc_of(block->options->layout()->rank()));
  Box lo{c->kl(), c->jl(), c->il()}, hi{c->ku() + 1, c->ju() + 1, c->iu() + 1};
  auto a = c->x2v.to(w.options()).view({1, -1, 1});
  auto b = c->x3v.to(w.options()).view({-1, 1, 1});
  auto r = c->x1v.to(w.options()).view({1, 1, -1});
  auto bc = basis(a, b, r, panel);
  auto field = torch::zeros({7, w.size(1), w.size(2), w.size(3)}, w.options());
  field[0].copy_(temp);
  field.narrow(0, 1, 3).copy_(w.narrow(0, IVX, 3));
  cs_contra_to_cart_(field.narrow(0, 1, 3), a, b, panel);
  field.narrow(0, 4, 3).copy_(r * bc[0]);
  auto cells = sub(field, lo, hi);
  auto mu = options->dynamic() ? torch::full_like(temp, options->nu_iso())
                               : options->nu_iso() * w[IDN];
  auto k = options->dynamic() ? torch::full_like(temp, options->kappa_iso())
                              : options->kappa_iso() * w[IDN] *
                                    phydro->peos->specific_heat_cv(w, temp);
  if (nu_scale_w_.defined()) mu.mul_(nu_scale_w_.view({1, 1, -1}));
  if (kappa_scale_w_.defined()) k.mul_(kappa_scale_w_.view({1, 1, -1}));
  auto coefficients = sub(torch::stack({mu, k}), lo, hi);
  std::array<double, 3> h{op->dx1(), op->dx2(), op->dx3()};
  std::array<Tensor, 3> flux, trace, normals;
  for (int dir = 0; dir < 3; ++dir) {
    int dim = 2 - dir, axis = dim - 3, n = hi[dim] - lo[dim];
    auto interp = weights(n, true, false, h[dir], w);
    auto face = transform(cells, interp, axis);
    std::array<Tensor, 3> deriv;
    for (int j = 0; j < 3; ++j) {
      deriv[j] =
          j == dir
              ? transform(cells, weights(n, true, true, h[j], w), axis)
              : transform(face,
                          weights(hi[2 - j] - lo[2 - j], false, true, h[j], w),
                          -1 - j);
    }
    Box fl = lo, fh = hi;
    ++fh[dim];
    auto af = (dir == 1 ? c->x2f : c->x2v)
                  .to(w.options())
                  .slice(0, fl[1], fh[1])
                  .view({1, -1, 1});
    auto bf = (dir == 2 ? c->x3f : c->x3v)
                  .to(w.options())
                  .slice(0, fl[0], fh[0])
                  .view({-1, 1, 1});
    auto rf = (dir == 0 ? c->x1f : c->x1v)
                  .to(w.options())
                  .slice(0, fl[2], fh[2])
                  .view({1, 1, -1});
    auto bases = basis(af, bf, rf, panel);
    auto normal = bases[0];
    if (dir > 0) {
      auto cos = vector_dot(bases[1], bases[2]);
      normal =
          (bases[dir] - cos * bases[3 - dir]) / torch::sqrt(1. - cos * cos);
    }
    // Differentiate Cartesian velocity and the Cartesian map with exactly
    // the same stencil: every affine Cartesian velocity has exact gradient,
    // including zero-strain solid rotation on every panel.
    // The reciprocal basis is the explicit 3x3 inverse. Besides avoiding
    // general LU solves, this avoids concurrent lazy CUDA linalg loading in
    // block workers. All operations stay on the calling worker's stream.
    std::array<Tensor, 3> reciprocal;
    for (int j = 0; j < 3; ++j)
      reciprocal[j] = torch::cross(deriv[(j + 1) % 3].narrow(0, 4, 3),
                                   deriv[(j + 2) % 3].narrow(0, 4, 3), 0);
    auto determinant = vector_dot(deriv[0].narrow(0, 4, 3), reciprocal[0]);
    auto vg =
        deriv[0].narrow(0, 1, 3).unsqueeze(1) * reciprocal[0].unsqueeze(0);
    for (int j = 1; j < 3; ++j)
      vg += deriv[j].narrow(0, 1, 3).unsqueeze(1) * reciprocal[j].unsqueeze(0);
    vg /= determinant;
    // Interpolating a curved map does not land exactly on the geometric face.
    // Correct that displacement so affine Cartesian velocity traces agree
    // across panels (in particular, rigid rotation creates no jump penalty).
    face.narrow(0, 1, 3).add_(
        (vg * (rf * bases[0] - face.narrow(0, 4, 3)).unsqueeze(0)).sum(1));
    // Linear coefficient reconstruction is positive in the interior. Physical
    // wall extrapolation falls back to the nearest cell if it is negative.
    auto cf = torch::empty({2, face.size(1), face.size(2), face.size(3)},
                           w.options());
    cf.narrow(axis, 1, n - 1)
        .copy_(0.5 * (coefficients.narrow(axis, 0, n - 1) +
                      coefficients.narrow(axis, 1, n - 1)));
    for (int side = 0; side < 2; ++side) {
      int near = side ? n - 1 : 0, next = side ? n - 2 : 1,
          index = side ? n : 0;
      auto v = coefficients.select(axis, near);
      auto extrap = 1.5 * v - 0.5 * coefficients.select(axis, next);
      cf.select(axis, index).copy_(torch::where(extrap >= 0., extrap, v));
    }
    auto div = vg[0][0] + vg[1][1] + vg[2][2];
    auto stress = cf[0] * (vg + vg.transpose(0, 1));
    for (int j = 0; j < 3; ++j) stress[j][j] -= (2. / 3.) * cf[0] * div;
    auto traction = (stress * normal.unsqueeze(0)).sum(1);
    Tensor heat;
    if (dir == 0) {
      // Radial resistance preserves T=A+B/r exactly for centre samples.
      auto rc = c->x1v.to(w.options()).slice(0, lo[2], hi[2]);
      heat = torch::zeros_like(face[0]);
      auto tl = cells[0].narrow(-1, 0, n - 1),
           tr = cells[0].narrow(-1, 1, n - 1);
      auto rl = rc.slice(0, 0, n - 1), rr = rc.slice(0, 1, n);
      heat.narrow(-1, 1, n - 1)
          .copy_(-cf[1].narrow(-1, 1, n - 1) * (tr - tl) * rl * rr / (rr - rl) /
                 rf.narrow(-1, 1, n - 1).square());
      for (int side = 0; side < 2; ++side) {
        int index = side ? n : 0, ic = side ? n - 1 : 0;
        double tw =
            side ? options->temperature_outer() : options->temperature_inner();
        auto wallr = rf.select(-1, index), walln = normal.select(-1, index);
        auto wallt = traction.select(-1, index);
        auto vn = vector_dot(wallt, walln);
        if (options->wall_velocity() == "stress-free")
          wallt.copy_(walln * vn);
        else {
          auto slope = cells.narrow(0, 1, 3).select(-1, ic) / (rc[ic] - wallr);
          wallt.copy_(cf[0].select(-1, index) *
                      (slope + walln * vector_dot(walln, slope) / 3.));
        }
        face.narrow(0, 1, 3).select(-1, index).zero_();
        if (tw > 0.)
          heat.select(-1, index).copy_(
              -cf[1].select(-1, index) * (cells[0].select(-1, ic) - tw) /
              ((rc[ic] - wallr) / (wallr * rc[ic])) / wallr.square());
      }
    } else {
      // Analytic angular metric. No radial derivative can contaminate a
      // horizontal heat flux, even on a strongly stratified thin shell.
      auto x = af.tan(), y = bf.tan(), s = 1. + x * x + y * y;
      auto gaa = s / (1. + x * x), gbb = s / (1. + y * y),
           gab = s * x * y / ((1. + x * x) * (1. + y * y));
      auto gnn = dir == 1 ? gaa : gbb;
      heat =
          -cf[1] / rf *
          (gnn.sqrt() * deriv[dir][0] + gab / gnn.sqrt() * deriv[3 - dir][0]);
    }
    if (dir == 0 && options->implicit_radial()) heat.zero_();
    flux[dir] = torch::zeros({5, w.size(1), w.size(2), w.size(3)}, w.options());
    auto f = sub(flux[dir], fl, fh);
    f.narrow(0, IVX, 3).copy_(-traction);
    f[IPR].copy_(heat - vector_dot(face.narrow(0, 1, 3), traction));
    trace[dir] =
        torch::zeros({7, w.size(1), w.size(2), w.size(3)}, w.options());
    auto t = sub(trace[dir], fl, fh);
    t.narrow(0, 0, 4).copy_(face.narrow(0, 0, 4));
    t.narrow(0, 4, 2).copy_(cf);
    // Only the boundary values of distance are exchanged.
    for (int side = 0; side < 2; ++side) {
      auto xc = cells.narrow(0, 4, 3).select(axis, side ? n - 1 : 0);
      auto xf = (rf * bases[0]).select(axis, side ? n : 0);
      t[6].select(axis, side ? n : 0)
          .copy_(vector_dot(xf - xc, normal.select(axis, side ? n : 0)).abs());
    }
    normals[dir] =
        torch::zeros({3, w.size(1), w.size(2), w.size(3)}, w.options());
    sub(normals[dir], fl, fh).copy_(normal);
  }
  share_fluxes(block, flux, trace, normals);
  auto tendency = -c->divergence(flux[0], flux[1], flux[2]);
  auto momentum = tendency.narrow(0, IVX, 3);
  cs_cart_to_contra_(momentum, a, b, panel);
  coord_vec_lower_(momentum, c->cosine_cell_kj.to(w.options()));
  du.narrow(0, 0, 5).add_(dt * tendency);
  return du;
}
}  // namespace snap
