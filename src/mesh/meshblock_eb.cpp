// Static Cartesian embedded boundaries and the halo of provisional states
// consumed by weighted state redistribution.
#include <kintera/constants.h>

#include <kintera/thermo/thermo.hpp>
#include <snap/eb/redistribution.hpp>
#include <snap/eb/sphere_exchange.hpp>
#include <snap/eb/sphere_transport.hpp>
#include <snap/eos/equation_of_state.hpp>
#include <snap/mesh/meshblock.hpp>

namespace snap {
namespace {
using Tensor = torch::Tensor;
using Bounds = std::array<int64_t, 3>;
// One receiving edge, two reconstruction layers, and three layers for
// the members, their overlap counts, and neighboring donor geometry.
constexpr int halo = 6;
constexpr int first_centroid = 14;
constexpr int valid_cell = 32;

Tensor box(Tensor t, Bounds const& lo, Bounds const& hi) {
  for (int d = 0; d < 3; ++d) t = t.slice(-1 - d, lo[d], hi[d]);
  return t;
}

// Exchange successive layers, including previously received transverse
// layers. This also handles a block narrower than the stencil and corners.
Tensor extend(MeshBlockImpl const& block, Tensor t, std::array<int, 3> blocks,
              bool geometry, std::array<double, 3> const& periods) {
  Exchange ex(&block);
  auto layout = block.get_layout();
  auto [x2, x3, x1] = layout->loc_of(layout->options->rank());
  std::array<int, 3> loc{x1, x2, x3};
  for (int d = 0; d < 3; ++d) {
    if (blocks[d] == 1) continue;
    int ax = t.dim() - 1 - d;
    auto n = t.size(ax);
    for (int depth = 0; depth < halo; ++depth) {
      // Send the next unsent cell toward each peer.
      auto low = t.narrow(ax, 2 * depth, 1).contiguous();
      auto high = t.narrow(ax, n - 1, 1).contiguous();
      int tag = 21300 + 32 * d + 4 * depth;
      auto peers = d == 0 ? ex.x1_seam_slab(low, high, tag, false)
                          : ex.seam_slab(d + 1, low, high, tag, false);
      auto receive = [&](Tensor value, bool lower) {
        if (!value.defined()) {
          value = torch::zeros_like(low);
          if (geometry)
            value[1].fill_(
                1);  // exterior: covered, with a valid background volume
        } else if (geometry && periods[d] &&
                   loc[d] == (lower ? 0 : blocks[d] - 1)) {
          value = value.clone();
          double shift = lower ? -periods[d] : periods[d];
          value[2 + d].add_(shift);
          for (int face = 0; face < 6; ++face)
            value[first_centroid + 3 * face + d].add_(shift);
        }
        return value;
      };
      t = torch::cat(
          {receive(peers.first, true), t, receive(peers.second, false)}, ax);
    }
  }
  return t;
}
}  // namespace

class EmbeddedBoundaryUpdate {
 public:
  explicit EmbeddedBoundaryUpdate(MeshBlockImpl const& block) {
    auto const& eb = *block.eb_geometry();
    auto c = block.pcoord;
    auto const& l = *block.options->layout();
    blocks_ = {l.pz(), l.px(), l.py()};
    std::array<bool, 3> periodic{
        l.periodic_z() || Exchange(&block).kind(0) == FaceKind::periodic,
        l.periodic_x(), l.periodic_y()};
    auto co = c->options;
    periods_ = {co->global_x1max() - co->global_x1min(),
                co->global_x2max() - co->global_x2min(),
                co->global_x3max() - co->global_x3min()};
    for (int d = 0; d < 3; ++d) {
      if (!periodic[d]) periods_[d] = 0;
      periodic[d] = periodic[d] && blocks_[d] == 1;
      int n = eb.volume_fraction.size(-1 - d);
      lo_[d] = n == 1 ? 0 : co->nghost();
      hi_[d] = n - lo_[d];
      inner_lo_[d] = blocks_[d] == 1 ? 0 : halo;
      inner_hi_[d] = inner_lo_[d] + hi_[d] - lo_[d];
    }
    auto eos = block.phydro->peos;
    hydro_rows_ = eos->nvar();
    const int constituents = 1 + hydro_rows_ - ICY;
    auto basis =
        torch::zeros({hydro_rows_, 1, 1, constituents}, c->x1f.options());
    basis[IDN].flatten()[0] = 1;
    for (int i = 1; i < constituents; ++i) basis[ICY + i - 1].flatten()[i] = 1;
    auto tmin = torch::full(
        {1, 1, constituents},
        eos->options->limiter() ? eos->options->temperature_floor() : 1.e-12,
        basis.options());
    min_energy_ = eos->compute("UT->I", {basis, tmin}).flatten();
    if (eb.panel >= 0) {
      sphere_ = std::make_unique<SphereExchange>(block);
      std::vector<Tensor> fields{eb.volume_fraction, eb.background_volume};
      for (int d = 0; d < 3; ++d) fields.push_back(eb.cell_centroid[d]);
      for (int d = 0; d < 3; ++d) fields.push_back(eb.wall_area_vector[d]);
      for (int side = 0; side < 6; ++side) {
        int d = side / 2, start = side % 2, n = eb.volume_fraction.size(-1 - d);
        auto at_side = [&](Tensor t) { return t.narrow(-1 - d, start, n); };
        fields.push_back(at_side(eb.face_fraction[d]));
        for (int k = 0; k < 3; ++k)
          fields.push_back(at_side(eb.face_centroid[d][k]));
        double sign = start ? 1. : -1.;
        for (int k = 0; k < 3; ++k)
          fields.push_back(sign * at_side(eb.face_area_vector[d][k]));
      }
      auto global = sphere_->gather(box(torch::stack(fields), lo_, hi_));
      int64_t n = global.size(1);
      auto sides = global.slice(0, 8).reshape({6, 7, n});
      auto adj = sphere_->adjacency().clone();
      auto back = sphere_->back_faces();
      auto reciprocal = (back.clamp_min(0) * n + adj.clamp_min(0)).flatten();
      // Compare vectors on their physical scale: a component can vanish by
      // symmetry even when the full face has a planetary radius or area.
      auto peer = [&](Tensor local) {
        return local.flatten().index_select(0, reciprocal).reshape_as(local);
      };
      auto valid = adj.ge(0);
      auto fraction = sides.select(1, 0);
      auto open = fraction.gt(0);
      double tol = global.scalar_type() == torch::kFloat64 ? 5.e-10 : 8.e-6;
      auto different =
          (open != peer(open)) | ((fraction - peer(fraction)).abs() > tol);
      TORCH_CHECK(!(different & valid).any().item<bool>(),
                  "cubed-sphere open apertures disagree across owned cells");
      for (int first : {1, 4}) {
        auto here = sides.narrow(1, first, 3);
        std::vector<Tensor> images;
        for (int k = 0; k < 3; ++k) images.push_back(peer(here.select(1, k)));
        auto there = torch::stack(images, 1);
        if (first == 4)
          there = -there;  // outward areas have opposite orientation
        auto scale = torch::maximum(here.norm(2, 1), there.norm(2, 1));
        auto mismatch = (here - there).norm(2, 1) > tol * scale;
        TORCH_CHECK(!(mismatch & valid & open).any().item<bool>(),
                    "cubed-sphere ", first == 1 ? "centroids" : "vector areas",
                    " disagree across owned faces");
      }
      adj.masked_fill_(~open, -1);
      EBGeometry packed;
      packed.volume_fraction = global[0].reshape({1, 1, n});
      packed.cell_centroid = global.slice(0, 2, 5).reshape({3, 1, 1, n});
      packed.wall_area_vector = global.slice(0, 5, 8).reshape({3, 1, 1, n});
      op_ = std::make_unique<WeightedStateRedistribution>(
          packed, global[1], std::array<bool, 3>{}, std::array<double, 3>{},
          Bounds{0, 0, 0}, Bounds{-1, -1, -1}, adj);
      return;
    }
    auto background =
        c->dx3f.view({-1, 1, 1}) * c->dx2f.view({1, -1, 1}) * c->dx1f;
    std::vector<Tensor> channels{eb.volume_fraction, background};
    for (int d = 0; d < 3; ++d) channels.push_back(eb.cell_centroid[d]);
    for (int d = 0; d < 3; ++d) channels.push_back(eb.wall_area_vector[d]);
    for (int d = 0; d < 3; ++d) {
      auto face = eb.face_fraction[d];
      auto n = eb.volume_fraction.size(-1 - d);
      channels.push_back(face.narrow(-1 - d, 0, n));
      channels.push_back(face.narrow(-1 - d, 1, n));
    }
    for (int d = 0; d < 3; ++d) {
      auto n = eb.volume_fraction.size(-1 - d);
      for (int side = 0; side < 2; ++side)
        for (int comp = 0; comp < 3; ++comp)
          channels.push_back(eb.face_centroid[d][comp].narrow(-1 - d, side, n));
    }
    channels.push_back(torch::ones_like(eb.volume_fraction));
    auto data = extend(block, box(torch::stack(channels), lo_, hi_), blocks_,
                       true, periods_);
    EBGeometry g;
    g.volume_fraction = data[0];
    g.cell_centroid = data.slice(0, 2, 5);
    g.wall_area_vector = data.slice(0, 5, 8);
    for (int d = 0; d < 3; ++d) {
      auto left = data[8 + 2 * d], right = data[9 + 2 * d];
      auto n = left.size(-1 - d);
      // At a physical exterior the packed padding cells have zero aperture.
      // The domain-side face is still allowed to carry an exterior BC flux.
      auto adjacent = data[valid_cell].narrow(-1 - d, 0, n - 1).ne(0) &
                      data[valid_cell].narrow(-1 - d, 1, n - 1).ne(0);
      auto compare = [&](Tensor lower, Tensor upper, Tensor mask) {
        TORCH_CHECK(
            torch::allclose(upper.narrow(-1 - d, 0, n - 1).masked_select(mask),
                            lower.narrow(-1 - d, 1, n - 1).masked_select(mask),
                            1.e-8, 1.e-12),
            "embedded boundary metrics disagree at shared faces in x", d + 1);
      };
      compare(left, right, adjacent);
      auto faces = [&](Tensor lower, Tensor upper) {
        return torch::cat({lower, upper.narrow(-1 - d, n - 1, 1)}, -1 - d);
      };
      g.face_fraction[d] = faces(left, right);
      std::vector<Tensor> coords;
      for (int comp = 0; comp < 3; ++comp) {
        int i = first_centroid + 6 * d + comp;
        compare(data[i], data[i + 3],
                adjacent & right.narrow(-1 - d, 0, n - 1).gt(0));
        coords.push_back(faces(data[i], data[i + 3]));
      }
      g.face_centroid[d] = torch::stack(coords);
    }
    op_ = std::make_unique<WeightedStateRedistribution>(
        g, data[1], periodic, periods_, inner_lo_, inner_hi_);
  }
  Tensor apply(MeshBlockImpl const& block, Tensor state) const {
    auto owned = box(state, lo_, hi_);
    if (sphere_) {
      auto work = state.clone();
      work.narrow(0, IVX, 3).copy_(
          sphere_momentum_to_cart(block, state.narrow(0, IVX, 3)));
      auto global = sphere_->gather(box(work, lo_, hi_));
      Tensor bad;
      auto result = op_->apply(global.reshape({global.size(0), 1, 1, -1}),
                               hydro_rows_, &bad, min_energy_);
      auto updated = sphere_->take(result.flatten(1));
      // Only rotate the increment back. Identity rows thus remain bitwise
      // unchanged, including regular cells on panels far from the boundary.
      auto difference = updated - box(work, lo_, hi_);
      auto delta = torch::zeros_like(state);
      box(delta, lo_, hi_).copy_(difference);
      delta.narrow(0, IVX, 3).copy_(
          sphere_cart_to_momentum(block, delta.narrow(0, IVX, 3)));
      owned.add_(box(delta, lo_, hi_));
      return bad;
    }
    auto full = extend(block, owned, blocks_, false, periods_);
    Tensor bad;
    owned.copy_(box(op_->apply(full, hydro_rows_, &bad, min_energy_), inner_lo_,
                    inner_hi_));
    return bad;
  }

 private:
  std::unique_ptr<SphereExchange> sphere_;
  int hydro_rows_;
  Tensor min_energy_;
  Bounds lo_, hi_, inner_lo_, inner_hi_;
  std::array<int, 3> blocks_;
  std::array<double, 3> periods_;
  std::unique_ptr<WeightedStateRedistribution> op_;
};

void MeshBlockImpl::set_embedded_boundary(EBGeometry const& g) {
  bool sphere = g.panel >= 0;
  auto coord_type = options->coord()->type();
  TORCH_CHECK(
      sphere ? (coord_type == "gnomonic-equiangle" &&
                options->layout()->type() == "cubed-sphere" &&
                g.panel == std::get<2>(
                               get_layout()->loc_of(options->layout()->rank())))
             : coord_type == "cartesian",
      "embedded geometry must match the block coordinate system and panel");
  TORCH_CHECK(!phydro->options->icorr() &&
                  !phydro->options->perturbation_flux() && !phydro->anomaly(),
              "embedded boundaries require explicit total-state dynamics");
  std::array<bool, 3> disabled{phydro->options->disable_flux_x1(),
                               phydro->options->disable_flux_x2(),
                               phydro->options->disable_flux_x3()};
  if (sphere) {
    auto coord = pcoord->options;
    std::array<int, 3> cells{coord->global_nx1(), coord->global_nx2(),
                             coord->global_nx3()};
    for (int d = 0; d < 3; ++d)
      TORCH_CHECK(
          !disabled[d] || cells[d] == 1,
          "mapped embedded boundaries require flux in resolved direction x",
          d + 1);
    TORCH_CHECK(
        phydro->options->conserve_axial_momentum() == "off" &&
            !phydro->flux_covariance() && !phydro->x1_mass_covariance(),
        "mapped embedded boundaries require conserve-axial-momentum: off, "
        "flux-covariance: false and x1-mass-covariance: false");
  }
  auto type = phydro->peos->options->type();
  TORCH_CHECK(
      type == "ideal-gas" || type == "ideal-moist" || type == "moist-mixture",
      "unsupported embedded-boundary EOS: ", type);
  auto gravity = phydro->options->grav();
  TORCH_CHECK(!gravity || (gravity->grav1() == 0 && gravity->grav2() == 0 &&
                           gravity->grav3() == 0),
              "embedded-boundary transport with gravity requires cut-cell "
              "potential-energy accounting");
  TORCH_CHECK(
      !phydro->options->diffusion(),
      "embedded-boundary diffusive face and wall fluxes are unsupported");
  TORCH_CHECK(!options->execution_strict(),
              "embedded boundaries require eager stage execution");
  auto c = pcoord;
  _eb = std::make_shared<EBGeometry>(
      (sphere ? EBGeometry::from_cubed_sphere_metrics(
                    g.volume_fraction, g.cell_centroid, g.face_fraction,
                    g.face_centroid, g.face_area_vector, g.wall_area_vector,
                    g.wall_centroid, c->x1f, c->x2f, c->x3f, g.panel)
              : EBGeometry::from_metrics(g.volume_fraction, g.cell_centroid,
                                         g.face_fraction, g.face_centroid,
                                         g.wall_area_vector, g.wall_centroid,
                                         c->x1f, c->x2f, c->x3f))
          .to(c->x1f.device()));
  for (int d = 0; d < 3; ++d) {
    if (disabled[d] || g.volume_fraction.size(-1 - d) == 1) {
      auto af = _eb->face_fraction[d];
      TORCH_CHECK(torch::allclose(af.slice(-1 - d, 0, -1), af.slice(-1 - d, 1),
                                  1.e-9, 1.e-12),
                  "embedded wall varies along inactive flux direction x",
                  d + 1);
    }
  }
  auto owned = torch::zeros_like(_eb->flags, torch::kBool);
  auto sub = part({0, 0, 0}, PartOptions().exterior(false).ndim(3));
  owned.index(sub).copy_(_eb->volume_fraction.index(sub) > 0);
  _eb_fluid_indices = owned.flatten().nonzero().flatten();
  auto const& eos = phydro->peos->options;
  auto const& th = eos->thermo();
  double temp = std::max(th ? th->Tref() : 300., 2 * eos->temperature_floor());
  double rd = kintera::constants::Rgas / eos->weight();
  double rho = std::max({(th ? th->Pref() : 1.e5) / (rd * temp),
                         2 * eos->density_floor(),
                         2 * eos->pressure_floor() / (rd * temp)});
  _eb_neutral_w =
      torch::zeros({phydro->peos->nvar(), 1, 1, 1}, c->x1f.options());
  _eb_neutral_w[IDN].fill_(rho);
  _eb_neutral_w[IPR].fill_(rho * rd * temp);
  _eb_neutral_u = _eb_neutral_w.clone();
  _eb_neutral_u[IPR].copy_(phydro->peos->compute("W->I", {_eb_neutral_w}));
  _eb_update.reset();
  bump_config_generation();
}

void MeshBlockImpl::prepare_embedded_boundary() {
  if (_eb && !_eb_update)
    _eb_update = std::make_shared<EmbeddedBoundaryUpdate>(*this);
}

Tensor MeshBlockImpl::redistribute(Tensor hydro, Tensor scalar) {
  prepare_embedded_boundary();
  if (!scalar.defined()) {
    return _eb_update->apply(*this, hydro);
  }
  auto state = torch::cat({hydro, scalar}, 0);
  auto bad = _eb_update->apply(*this, state);
  hydro.copy_(state.narrow(0, 0, hydro.size(0)));
  scalar.copy_(state.narrow(0, hydro.size(0), scalar.size(0)));
  return bad;
}

void MeshBlockImpl::write_eb_restart(Variables& data) const {
  if (!_eb) return;
  auto const& g = *_eb;
  data["eb.volume_fraction"] = g.volume_fraction.cpu();
  if (g.panel >= 0) data["eb.panel"] = torch::tensor(g.panel, torch::kInt64);
  data["eb.cell_centroid"] = g.cell_centroid.cpu();
  data["eb.wall_area_vector"] = g.wall_area_vector.cpu();
  data["eb.wall_centroid"] = g.wall_centroid.cpu();
  for (int d = 0; d < 3; ++d) {
    auto i = std::to_string(d + 1);
    if (g.panel >= 0) data["eb.face_vector" + i] = g.face_area_vector[d].cpu();
    data["eb.face_fraction" + i] = g.face_fraction[d].cpu();
    data["eb.face_centroid" + i] = g.face_centroid[d].cpu();
  }
}

void MeshBlockImpl::read_eb_restart(Variables const& data) {
  TORCH_CHECK(!data.count("solid"),
              "restart contains retired solid mask; initialize with embedded "
              "boundary geometry");
  if (!data.count("eb.volume_fraction")) {
    TORCH_CHECK(
        !_eb, "restart contains no embedded boundary geometry for this block");
    return;
  }
  EBGeometry g;
  if (data.count("eb.panel")) g.panel = data.at("eb.panel").item<int>();
  g.volume_fraction = data.at("eb.volume_fraction");
  g.cell_centroid = data.at("eb.cell_centroid");
  g.wall_area_vector = data.at("eb.wall_area_vector");
  g.wall_centroid = data.at("eb.wall_centroid");
  for (int d = 0; d < 3; ++d) {
    auto i = std::to_string(d + 1);
    if (g.panel >= 0) g.face_area_vector[d] = data.at("eb.face_vector" + i);
    g.face_fraction[d] = data.at("eb.face_fraction" + i);
    g.face_centroid[d] = data.at("eb.face_centroid" + i);
  }
  set_embedded_boundary(g);
}
}  // namespace snap
