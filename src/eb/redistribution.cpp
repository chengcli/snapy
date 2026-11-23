#include "redistribution.hpp"

#include <snap/snap.h>

#include <Eigen/QR>
#include <algorithm>
#include <cmath>
#include <map>
#include <set>
#include <utility>
#include <vector>

#include "../coord/x1_stencil.hpp"
#include "geometry.hpp"

namespace snap {
namespace {
using Index = int64_t;
using Indices = std::array<Index, 3>;
using Point = std::array<double, 3>;
using Row = std::vector<Index>;
using Weights = std::vector<double>;

torch::Tensor host(const torch::Tensor &value) {
  return value.to(torch::kCPU, torch::kFloat64).contiguous();
}

struct Grid {
  Indices size, lo, hi;
  std::array<bool, 3> periodic;
  Point periods;
  torch::Tensor fraction, volume, centroids, wall;
  std::array<torch::Tensor, 3> faces;

  Index id(Indices p) const { return (p[2] * size[1] + p[1]) * size[0] + p[0]; }

  Indices position(Index i) const {
    return {i % size[0], i / size[0] % size[1], i / (size[0] * size[1])};
  }

  bool owned(Index i) const {
    const auto p = position(i);
    for (int d = 0; d < 3; ++d)
      if (p[d] < lo[d] || p[d] >= hi[d]) return false;
    return true;
  }

  double value(const torch::Tensor &t, Index i) const {
    return t.const_data_ptr<double>()[i];
  }

  Point point(const torch::Tensor &t, Index i) const {
    Index n = fraction.numel();
    return {value(t, i), value(t, n + i), value(t, 2 * n + i)};
  }

  Point offset(Point a, Point b) const {
    for (int d = 0; d < 3; ++d) {
      a[d] -= b[d];
      if (periodic[d]) a[d] = std::remainder(a[d], periods[d]);
    }
    return a;
  }

  Index step(Index i, int axis, int sign) const {
    if (size[axis] == 1) return -1;
    auto cell = position(i);
    auto face = cell;
    if (sign > 0) ++face[axis];
    const auto &f = faces[axis];
    Index fi = (face[2] * f.size(1) + face[1]) * f.size(2) + face[0];
    if (value(f, fi) <= 0) return -1;

    cell[axis] += sign;
    if (cell[axis] < 0 || cell[axis] == size[axis]) {
      if (!periodic[axis]) return -1;
      cell[axis] = (cell[axis] + size[axis]) % size[axis];
    }
    Index j = id(cell);
    return value(fraction, j) > 0 ? j : -1;
  }

  // Reach only cells joined by open Cartesian faces inside the stencil box.
  Row connected(Index i, Indices lower, Indices upper) const {
    Row queue{i};
    std::set<Index> seen{i};
    const auto origin = position(i);
    for (size_t k = 0; k < queue.size(); ++k) {
      for (int axis = 0; axis < 3; ++axis) {
        for (int sign : {-1, 1}) {
          const Index j = step(queue[k], axis, sign);
          if (j < 0 || seen.count(j)) continue;
          Index dx = position(j)[axis] - origin[axis];
          bool inside = dx >= lower[axis] && dx <= upper[axis];
          if (periodic[axis]) {
            inside |= dx + size[axis] >= lower[axis] &&
                      dx + size[axis] <= upper[axis];
            inside |= dx - size[axis] >= lower[axis] &&
                      dx - size[axis] <= upper[axis];
          }
          if (inside) {
            seen.insert(j);
            queue.push_back(j);
          }
        }
      }
    }
    return queue;
  }

  Row neighborhood(Index i, double target) const {
    // Rank signed Cartesian axes by the open direction of the wall normal.
    // Try the opposite sign when the preferred side ends at a physical wall.
    Point normal = point(wall, i);
    std::vector<std::pair<double, int>> choices;
    for (int d = 0; d < 3; ++d) {
      if (size[d] == 1) continue;
      int sign = normal[d] > 0 ? -1 : 1;
      if (step(i, d, sign) < 0 && step(i, d, -sign) >= 0) sign = -sign;
      choices.emplace_back(std::abs(normal[d]), sign * (d + 1));
    }
    std::stable_sort(choices.begin(), choices.end(),
                     [](auto a, auto b) { return a.first > b.first; });
    Indices lower{}, upper{};
    Row row{i};
    for (size_t k = 0; k < choices.size(); ++k) {
      int direction = choices[k].second;
      int d = std::abs(direction) - 1;
      if (direction < 0)
        lower[d] = -1;
      else
        upper[d] = 1;
      row = connected(i, lower, upper);
      double total = 0;
      for (auto j : row) total += value(volume, j);
      bool tied = k + 1 < choices.size() && choices[k].first > 0 &&
                  std::abs(choices[k].first - choices[k + 1].first) <=
                      1e-12 * choices[k].first;
      if (total >= target && !tied) break;
    }
    return row;
  }
};

struct Neighborhood {
  Row cells;
  double beta;
};

struct Average {
  Row cells;
  Weights assignment, weight;
  Point centroid;
};

// Build a scaled least-squares inverse. A singular stencil signals that the
// caller should grow the stencil or use a constant reconstruction.
bool least_squares(const std::vector<Point> &offsets, Indices size,
                   std::array<Weights, 3> &coefficients) {
  std::vector<int> axes;
  for (int d = 0; d < 3; ++d)
    if (size[d] > 1) axes.push_back(d);
  Eigen::MatrixXd design(offsets.size(), axes.size());
  Point scale{};
  for (size_t col = 0; col < axes.size(); ++col) {
    int d = axes[col];
    for (size_t k = 0; k < offsets.size(); ++k) design(k, col) = offsets[k][d];
    scale[d] = design.col(col).cwiseAbs().maxCoeff();
    if (scale[d] == 0.) return false;
    design.col(col) /= scale[d];
  }
  auto qr = design.colPivHouseholderQr();
  qr.setThreshold(1e-10);
  if (qr.rank() != design.cols()) return false;
  Eigen::MatrixXd inverse =
      qr.solve(Eigen::MatrixXd::Identity(design.rows(), design.rows()));
  for (auto &row : coefficients) row.assign(offsets.size(), 0.);
  for (size_t col = 0; col < axes.size(); ++col)
    for (size_t k = 0; k < offsets.size(); ++k)
      coefficients[axes[col]][k] = inverse(col, k) / scale[axes[col]];
  return true;
}

Index width(const std::vector<Row> &rows) {
  size_t result = 1;
  for (auto const &row : rows) result = std::max(result, row.size());
  return static_cast<Index>(result);
}

// Pad an index row with its first index, keeping padding local and finite.
torch::Tensor indices(std::vector<Row> rows, Index columns,
                      torch::Device device) {
  for (auto &row : rows) row.resize(columns, row.front());
  return x1_stencil_index(rows, columns).to(device);
}

torch::Tensor gather(const torch::Tensor &state, const torch::Tensor &rows) {
  return state.index_select(1, rows.flatten())
      .reshape({state.size(0), rows.size(0), rows.size(1)});
}

// All functions below accept [components, cells, members] or
// [components,cells].
torch::Tensor mass(const torch::Tensor &u, int hydro_rows) {
  auto rho = u[IDN];
  if (hydro_rows > ICY) rho = rho + u.slice(0, ICY, hydro_rows).sum(0);
  return rho;
}

torch::Tensor thermal_margin(const torch::Tensor &u, int hydro_rows,
                             const torch::Tensor &minima) {
  auto rho = mass(u, hydro_rows);
  auto kinetic = 0.5 * u.narrow(0, IVX, 3).square().sum(0) /
                 torch::where(rho > 0, rho, 1.);
  auto margin = u[IPR] - kinetic;
  if (minima.defined()) {
    margin = margin - minima[0] * u[IDN];
    for (int i = ICY; i < hydro_rows; ++i)
      margin = margin - minima[1 + i - ICY] * u[i];
  }
  return margin;
}

torch::Tensor inadmissible(const torch::Tensor &u, int hydro_rows,
                           const torch::Tensor &minima) {
  return (~u.isfinite().all()) | (u[IDN] < 0).any() |
         (u.slice(0, ICY) < 0).any() | (mass(u, hydro_rows) <= 0).any() |
         (thermal_margin(u, hydro_rows, minima) <= 0).any();
}

// For f(theta) concave, its chord from f(0) to f(1) is a lower bound.
// Back away from a limiting root to leave room for storage rounding.
torch::Tensor chord_factor(const torch::Tensor &start, const torch::Tensor &end,
                           double safety) {
  auto crossed = end <= 0;
  auto denom = torch::where(crossed & (start > end), start - end, 1.);
  return torch::where(crossed & (start > end),
                      safety * (start / denom).clamp(0, 1), 1.);
}

torch::Tensor limit_members(const torch::Tensor &center,
                            torch::Tensor increment, int hydro_rows,
                            const torch::Tensor &minima, double safety) {
  auto end = center + increment;
  auto factor = chord_factor(center[IDN], end[IDN], safety);
  if (end.size(0) > ICY) {
    auto partials =
        chord_factor(center.slice(0, ICY), end.slice(0, ICY), safety);
    factor = torch::minimum(factor, partials.amin(0));
  }
  factor = torch::minimum(factor, chord_factor(mass(center, hydro_rows),
                                               mass(end, hydro_rows), safety));
  increment = increment * factor.amin(-1, true);
  // E - |m|^2/(2 rho) - sum(rho_s e_min,s) is concave for positive total rho.
  auto energy_factor = chord_factor(
      thermal_margin(center, hydro_rows, minima),
      thermal_margin(center + increment, hydro_rows, minima), safety);
  return center + increment * energy_factor.amin(-1, true);
}

}  // namespace

WeightedStateRedistribution::WeightedStateRedistribution(
    const EBGeometry &geometry, const torch::Tensor &background_volume,
    std::array<bool, 3> periodic, Point periods, Indices lo, Indices hi) {
  auto fraction = host(geometry.volume_fraction);
  TORCH_CHECK(fraction.dim() == 3,
              "EB volume fractions must have shape [z,y,x]");
  shape_ = fraction.sizes().vec();
  Indices dims{shape_[2], shape_[1], shape_[0]};
  auto bg = host(background_volume);
  if (bg.dim() == 4 && bg.size(0) == 1) bg = bg.squeeze(0);
  bg = bg.expand_as(fraction).contiguous();
  TORCH_CHECK(bg.isfinite().all().item<bool>() && bg.min().item<double>() > 0,
              "EB background cell volumes must be finite and positive");
  Grid grid{dims,
            lo,
            hi,
            periodic,
            periods,
            fraction,
            fraction * bg,
            host(geometry.cell_centroid),
            host(geometry.wall_area_vector),
            {}};
  std::array<torch::Tensor, 3> face_coords;
  for (int d = 0; d < 3; ++d) {
    if (grid.hi[d] == -1) grid.hi[d] = dims[d];
    TORCH_CHECK(lo[d] >= 0 && lo[d] < grid.hi[d] && grid.hi[d] <= dims[d],
                "invalid EB owned extent on axis ", d);
    grid.periodic[d] &= dims[d] > 1;
    if (grid.periodic[d])
      TORCH_CHECK(std::isfinite(periods[d]) && periods[d] > 0,
                  "periodic WSRD requires a physical period on axis ", d);
    grid.faces[d] = host(geometry.face_fraction[d]);
    face_coords[d] = host(geometry.face_centroid[d].select(0, d));
  }
  Row owned_fluid;
  std::map<Index, Neighborhood> merged;
  std::map<Index, Row> foreign;
  for (Index i = 0; i < fraction.numel(); ++i) {
    double vf = grid.value(fraction, i);
    if (vf > 0 && grid.owned(i)) owned_fluid.push_back(i);
    if (vf <= 0 || vf >= 0.5) continue;
    double target = 0.5 * grid.value(bg, i);
    Row cells = grid.neighborhood(i, target);
    double neighbor_volume = 0;
    for (size_t k = 1; k < cells.size(); ++k)
      neighbor_volume += grid.value(grid.volume, cells[k]);
    const double deficit = target - grid.value(grid.volume, i);
    TORCH_CHECK(!grid.owned(i) || neighbor_volume >= deficit * (1 - 1e-12),
                "WSRD cannot reach half a background cell volume at (x,y,z) = ",
                grid.position(i)[0], ",", grid.position(i)[1], ",",
                grid.position(i)[2], " using connected open faces");
    if (neighbor_volume == 0) continue;
    double beta = std::min(1., deficit / neighbor_volume);
    for (size_t k = 1; k < cells.size(); ++k) foreign[cells[k]].push_back(i);
    merged.emplace(i, Neighborhood{std::move(cells), beta});
  }
  fluid_ = torch::tensor(owned_fluid, torch::kInt64)
               .to(geometry.volume_fraction.device());
  if (merged.empty()) return;

  auto average = [&](Index i) {
    Average a;
    const auto m = merged.find(i);
    a.cells = m == merged.end() ? Row{i} : m->second.cells;
    double total_volume = 0;
    const auto origin = grid.point(grid.centroids, i);
    Point shift{};
    for (auto j : a.cells) {
      auto it = foreign.find(j);
      double count = 1 + (it == foreign.end() ? 0 : it->second.size());
      double w = 1;
      if (j != i)
        w = m->second.beta / count;
      else if (it != foreign.end()) {
        for (auto other : it->second) w -= merged.at(other).beta / count;
      }
      a.assignment.push_back(w);
      double volume_weight = w * grid.value(grid.volume, j);
      a.weight.push_back(volume_weight);
      total_volume += volume_weight;
      auto dx = grid.offset(grid.point(grid.centroids, j), origin);
      for (int d = 0; d < 3; ++d) shift[d] += volume_weight * dx[d];
    }
    for (auto &w : a.weight) w /= total_volume;
    for (int d = 0; d < 3; ++d)
      a.centroid[d] = origin[d] + shift[d] / total_volume;
    return a;
  };

  // Cache only neighborhoods that contribute to owned targets or supply a
  // slope. Interior regular cells bypass the stage entirely.
  std::set<Index> targets, owners;
  for (auto const &[i, m] : merged)
    for (auto j : m.cells)
      if (grid.owned(j)) targets.insert(j);
  for (auto j : targets) {
    owners.insert(j);
    auto it = foreign.find(j);
    if (it != foreign.end())
      owners.insert(it->second.begin(), it->second.end());
  }
  if (targets.empty()) return;
  std::map<Index, Index> lookup;
  std::vector<Average> averages;
  auto cache = [&](Index i) {
    auto [it, fresh] = lookup.emplace(i, averages.size());
    if (fresh) averages.push_back(average(i));
    return it->second;
  };
  Row owner_rows;
  std::vector<Row> slopes, q_cells, edge_rows;
  std::vector<Weights> q_weights, edge_weights;
  std::array<std::vector<Weights>, 3> ls, slope_offsets, member_offsets;
  std::map<Index, std::vector<std::pair<Index, double>>> contributions;
  Index max_members = 1;
  for (auto i : owners)
    max_members = std::max(max_members,
                           static_cast<Index>(averages[cache(i)].cells.size()));
  for (auto i : owners) {
    Index center = cache(i);
    Average const a =
        averages[center];  // cache() may reallocate averages below
    Index owner = static_cast<Index>(owner_rows.size());
    owner_rows.push_back(center);
    std::array<Weights, 3> offsets;
    for (size_t k = 0; k < a.cells.size(); ++k) {
      Index j = a.cells[k];
      Point dx = grid.offset(grid.point(grid.centroids, j), a.centroid);
      for (int d = 0; d < 3; ++d) offsets[d].push_back(dx[d]);
      if (targets.count(j))
        contributions[j].emplace_back(owner * max_members + k, a.assignment[k]);
    }
    for (int d = 0; d < 3; ++d)
      member_offsets[d].push_back(std::move(offsets[d]));

    Row slope_row{center};
    std::vector<Point> displacements(1);
    std::array<Weights, 3> coeff;
    for (int radius = 1; a.cells.size() > 1 && radius <= 2; ++radius) {
      Indices lower{-radius, -radius, -radius}, upper{radius, radius, radius};
      slope_row.clear();
      displacements.clear();
      Point reach{};
      for (auto j : grid.connected(i, lower, upper)) {
        Index q = cache(j);
        auto dx = grid.offset(averages[q].centroid, a.centroid);
        slope_row.push_back(q);
        displacements.push_back(dx);
        for (int d = 0; d < 3; ++d)
          reach[d] = std::max(reach[d], std::abs(dx[d]));
      }
      bool enough = true;
      const auto pos = grid.position(i);
      for (int d = 0; d < 3; ++d) {
        if (dims[d] == 1) continue;
        auto f = face_coords[d].accessor<double, 3>();
        auto high = pos;
        ++high[d];
        double h = f[high[2]][high[1]][high[0]] - f[pos[2]][pos[1]][pos[0]];
        enough &= reach[d] > 0.5 * h;
      }
      if (enough && least_squares(displacements, dims, coeff)) break;
    }
    slopes.push_back(std::move(slope_row));
    for (int d = 0; d < 3; ++d) {
      if (coeff[d].empty()) coeff[d].assign(displacements.size(), 0.);
      ls[d].push_back(std::move(coeff[d]));
      Weights coord;
      for (auto const &dx : displacements) coord.push_back(dx[d]);
      slope_offsets[d].push_back(std::move(coord));
    }
  }
  for (auto &a : averages) {
    q_cells.push_back(std::move(a.cells));
    q_weights.push_back(std::move(a.weight));
  }
  for (auto j : targets) {
    Row rows;
    Weights w;
    for (auto const &[edge, weight] : contributions.at(j)) {
      rows.push_back(edge);
      w.push_back(weight);
    }
    edge_rows.push_back(std::move(rows));
    edge_weights.push_back(std::move(w));
  }
  auto device = geometry.volume_fraction.device();
  // MPS has no float64 storage. Keep host coefficients at full precision on
  // the other backends and cast to the state accumulation dtype at application.
  auto options = torch::TensorOptions().device(device).dtype(
      device.is_mps() ? torch::kFloat32 : torch::kFloat64);
  auto weights = [&](const std::vector<Weights> &w, Index columns) {
    return x1_stencil_weight(w, columns).to(options);
  };
  auto q_width = width(q_cells), slope_width = width(slopes),
       edge_width = width(edge_rows);
  q_cells_ = indices(std::move(q_cells), q_width, device);
  q_weight_ = weights(q_weights, q_width);
  owner_rows_ = torch::tensor(owner_rows, torch::kInt64).to(device);
  slope_rows_ = indices(std::move(slopes), slope_width, device);
  for (int d = 0; d < 3; ++d) {
    ls_weight_[d] = weights(ls[d], slope_width);
    slope_offset_[d] = weights(slope_offsets[d], slope_width);
    member_offset_[d] = weights(member_offsets[d], max_members);
  }
  targets_ = torch::tensor(Row(targets.begin(), targets.end()), torch::kInt64)
                 .to(device);
  edge_rows_ = indices(std::move(edge_rows), edge_width, device);
  edge_weight_ = weights(edge_weights, edge_width);
}

torch::Tensor WeightedStateRedistribution::apply(
    const torch::Tensor &state, int hydro_rows, torch::Tensor *bad,
    const torch::Tensor &minimum_specific_internal_energy) const {
  TORCH_CHECK((state.dim() == 3 || state.dim() == 4) &&
                  state.is_floating_point() &&
                  state.sizes().slice(state.dim() - 3).vec() == shape_,
              "WSRD state must be floating point with geometry shape "
              "[components,z,y,x]");
  TORCH_CHECK(state.device() == fluid_.device(),
              "WSRD geometry and state must share a device");
  TORCH_CHECK(hydro_rows == 0 || (state.dim() == 4 && hydro_rows >= ICY &&
                                  hydro_rows <= state.size(0)),
              "WSRD hydro component count does not match its conserved state");
  TORCH_CHECK(
      !minimum_specific_internal_energy.defined() ||
          (minimum_specific_internal_energy.dim() == 1 &&
           minimum_specific_internal_energy.numel() == 1 + hydro_rows - ICY &&
           minimum_specific_internal_energy.device() == state.device()),
      "WSRD minimum energies require one entry per hydro species on the state "
      "device");
  auto flat = state.reshape({-1, shape_[0] * shape_[1] * shape_[2]});
  auto minima = minimum_specific_internal_energy;
  if (bad) *bad = torch::zeros({}, state.options().dtype(torch::kBool));
  if (!targets_.defined()) {
    if (hydro_rows && bad)
      *bad = inadmissible(flat.index_select(1, fluid_), hydro_rows, minima);
    return state;
  }
  auto values = gather(flat, q_cells_).to(q_weight_.scalar_type());
  auto avg = (values * q_weight_).sum(-1);
  auto center = avg.index_select(1, owner_rows_).unsqueeze(-1);
  auto delta = gather(avg, slope_rows_) - center;
  torch::Tensor fitted, reconstructed;
  for (int d = 0; d < 3; ++d) {
    auto gradient = (delta * ls_weight_[d]).sum(-1, true);
    auto slope = gradient * slope_offset_[d];
    auto member = gradient * member_offset_[d];
    if (d == 0) {
      fitted = slope;
      reconstructed = member;
    } else {
      fitted += slope;
      reconstructed += member;
    }
  }
  // Limit predictions at the neighboring weighted centroids. A single factor
  // per neighborhood preserves linear relationships among conserved species.
  double tol = q_weight_.scalar_type() == torch::kFloat64 ? 1e-12 : 1e-6;
  auto nonzero = fitted.abs() > tol * (delta.abs() + center.abs());
  auto safe = torch::where(nonzero, fitted, torch::ones_like(fitted));
  auto factor = torch::where(nonzero, (delta / safe).clamp(0, 1), 1.);
  auto increment = factor.amin({0, 2}, true) * reconstructed;
  auto limited = center + increment;
  if (hydro_rows) {
    double eps = state.scalar_type() == torch::kFloat64 ? 2.22e-16 : 1.20e-7;
    limited =
        limit_members(center, increment, hydro_rows, minima, 1. - 16. * eps);
    if (bad) *bad = inadmissible(center, hydro_rows, minima);
  }
  auto output = (gather(limited.flatten(1), edge_rows_) * edge_weight_).sum(-1);
  auto result = flat.index_copy(1, targets_, output.to(state.scalar_type()));
  if (hydro_rows && bad)
    *bad =
        *bad | inadmissible(result.index_select(1, fluid_), hydro_rows, minima);
  return result.reshape_as(state);
}
}  // namespace snap
