#include "geometry.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <vector>

namespace snap {
namespace {

using Tensor = torch::Tensor;
using Vec = std::array<double, 3>;
using Index = std::array<int64_t, 3>;

Vec operator+(Vec a, const Vec &b) {
  for (int d = 0; d < 3; ++d) a[d] += b[d];
  return a;
}
Vec operator-(Vec a, const Vec &b) {
  for (int d = 0; d < 3; ++d) a[d] -= b[d];
  return a;
}
Vec operator*(double s, Vec a) {
  for (auto &x : a) x *= s;
  return a;
}
Vec cross(const Vec &a, const Vec &b) {
  return {a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2],
          a[0] * b[1] - a[1] * b[0]};
}
double dot(const Vec &a, const Vec &b) {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

struct Moments {
  double measure{};
  Vec first{};
  void add(double w, const Vec &centroid) {
    measure += w;
    first = first + w * centroid;
  }
  Vec centroid(const Vec &fallback) const {
    if (measure == 0) return fallback;
    // Division componentwise also works for subnormal measures.
    return {first[0] / measure, first[1] / measure, first[2] / measure};
  }
  void triangle(const Vec &a, const Vec &b, const Vec &c) {
    const auto n = cross(b - a, c - a);
    add(std::hypot(n[0], n[1], n[2]) / 2, (1.0 / 3) * (a + b + c));
  }
  void tet(const Vec &a, const Vec &b, const Vec &c, const Vec &d) {
    add(std::abs(dot(b - a, cross(c - a, d - a))) / 6, 0.25 * (a + b + c + d));
  }

  // Bottom abc and top ABC, with edges aA, bB, cC.
  void prism(const Vec &a, const Vec &b, const Vec &c, const Vec &A,
             const Vec &B, const Vec &C) {
    tet(a, b, c, A);
    tet(A, b, c, B);
    tet(A, c, B, C);
  }
};

struct Vertex {
  Vec p;
  double phi;
};
Vec crossing(const Vertex &a, const Vertex &b) {
  // Interpolate from the vertex nearest zero to resolve small fluid slivers.
  if (std::abs(a.phi) > std::abs(b.phi)) return crossing(b, a);
  const double ratio = std::abs(a.phi / b.phi);
  return a.p + (ratio / (1 + ratio)) * (b.p - a.p);
}

void clip_triangle(const std::array<Vertex, 3> &tri, Moments &face) {
  std::array<Vec, 4> polygon;
  int count = 0;
  for (int j = 0; j < 3; ++j) {
    const auto &a = tri[j];
    const auto &b = tri[(j + 1) % 3];
    if (a.phi > 0) polygon[count++] = a.p;
    if ((a.phi > 0) != (b.phi > 0)) polygon[count++] = crossing(a, b);
  }
  for (int j = 2; j < count; ++j)
    face.triangle(polygon[0], polygon[j - 1], polygon[j]);
}

void clip_tet(const std::array<Vertex, 4> &tet, Moments &volume,
              Moments &wall) {
  std::array<Vertex, 4> in{}, out{};
  int n = 0, m = 0;
  for (const auto &v : tet) {
    if (v.phi > 0)
      in[n++] = v;
    else
      out[m++] = v;
  }
  if (n == 0) return;
  if (n == 4) {
    volume.tet(in[0].p, in[1].p, in[2].p, in[3].p);
    return;
  }
  if (n == 1) {
    auto p = crossing(in[0], out[0]), q = crossing(in[0], out[1]),
         r = crossing(in[0], out[2]);
    volume.tet(in[0].p, p, q, r);
    wall.triangle(p, q, r);
    return;
  }
  if (n == 3) {
    auto p = crossing(in[0], out[0]), q = crossing(in[1], out[0]),
         r = crossing(in[2], out[0]);
    volume.prism(in[0].p, in[1].p, in[2].p, p, q, r);
    wall.triangle(p, q, r);
    return;
  }
  auto p = crossing(in[0], out[0]), q = crossing(in[0], out[1]),
       r = crossing(in[1], out[0]), s = crossing(in[1], out[1]);
  volume.prism(in[0].p, p, q, in[1].p, r, s);
  wall.triangle(p, q, r);
  wall.triangle(q, r, s);
}

Tensor host(const Tensor &t) {
  TORCH_CHECK(t.defined() && t.is_floating_point(),
              "EB metrics need floating point tensors");
  return t.detach().to(torch::kCPU, torch::kFloat64).contiguous();
}

struct Grid {
  std::array<Tensor, 3> axis;
  Index shape;
  Grid(const Tensor &x, const Tensor &y, const Tensor &z)
      : axis{host(x), host(y), host(z)} {
    for (int d = 0; d < 3; ++d) {
      TORCH_CHECK(axis[d].dim() == 1 && axis[d].numel() >= 2,
                  "EB coordinates must be one-dimensional face vectors");
      shape[d] = axis[d].numel() - 1;
      const auto *f = axis[d].const_data_ptr<double>();
      TORCH_CHECK(std::isfinite(f[0]), "non-finite EB coordinate");
      for (int64_t j = 1; j <= shape[d]; ++j)
        TORCH_CHECK(std::isfinite(f[j]) && f[j] > f[j - 1],
                    "EB face coordinates must increase strictly");
    }
  }
  Vec lower(const Index &i) const {
    return {axis[0].const_data_ptr<double>()[i[0]],
            axis[1].const_data_ptr<double>()[i[1]],
            axis[2].const_data_ptr<double>()[i[2]]};
  }
  Vec width(const Index &i) const {
    return lower({i[0] + 1, i[1] + 1, i[2] + 1}) - lower(i);
  }
};

int64_t offset(const Index &i, const Index &shape) {
  return (i[2] * shape[1] + i[1]) * shape[0] + i[0];
}
std::vector<int64_t> dims(const Index &shape, bool vector = false) {
  std::vector<int64_t> result{shape[2], shape[1], shape[0]};
  if (vector) result.insert(result.begin(), 3);
  return result;
}
template <class F>
void each(const Index &shape, F f) {
  for (int64_t k = 0; k < shape[2]; ++k)
    for (int64_t j = 0; j < shape[1]; ++j)
      for (int64_t i = 0; i < shape[0]; ++i) f(Index{i, j, k});
}
void write_vec(Tensor &t, int64_t i, const Vec &v) {
  const int64_t stride = t.numel() / 3;
  for (int d = 0; d < 3; ++d) t.data_ptr<double>()[d * stride + i] = v[d];
}
Vec read_vec(const Tensor &t, int64_t i) {
  const auto *p = t.const_data_ptr<double>() + i;
  const int64_t n = t.numel() / 3;
  return {p[0], p[n], p[2 * n]};
}
Index stagger(Index shape, int d) {
  shape[d]++;
  return shape;
}

EBGeometry allocate(const Index &shape) {
  EBGeometry g;
  g.volume_fraction = torch::empty(dims(shape), torch::kFloat64);
  g.cell_centroid = torch::empty(dims(shape, true), torch::kFloat64);
  g.wall_area_vector = torch::empty_like(g.cell_centroid);
  g.wall_centroid = torch::empty_like(g.cell_centroid);
  g.flags = torch::empty(dims(shape), torch::kUInt8);
  for (int d = 0; d < 3; ++d) {
    auto fs = stagger(shape, d);
    g.face_fraction[d] = torch::empty(dims(fs), torch::kFloat64);
    g.face_centroid[d] = torch::empty(dims(fs, true), torch::kFloat64);
  }
  return g;
}

// Each tetrahedron shares the low-to-high diagonal of its Cartesian face
// with the adjacent cube. Explicitly reject ambiguous alternating signs on
// faces and disconnected corner sets instead of joining separate EB sheets.
void check_topology(const std::array<Vertex, 8> &v) {
  for (bool fluid : {true, false}) {
    int mask = 0;
    for (int b = 0; b < 8; ++b)
      if (fluid ? v[b].phi > 0 : v[b].phi < 0) mask |= 1 << b;
    if (!mask) continue;
    // Zero vertices can connect a single sheet exactly on an edge.
    int pass = mask;
    for (int b = 0; b < 8; ++b)
      if (v[b].phi == 0) pass |= 1 << b;
    int seen = mask & -mask;
    for (int step = 0; step < 7; ++step)
      for (int b = 0; b < 8; ++b)
        if (seen & (1 << b))
          for (int d = 0; d < 3; ++d) seen |= pass & (1 << (b ^ (1 << d)));
    TORCH_CHECK((seen & mask) == mask,
                "disconnected EB components in one cell");
  }
}

Vec wall_vector(const EBGeometry &g, const Grid &grid, const Index &i,
                const Vec &h) {
  Vec w{};
  for (int d = 0; d < 3; ++d) {
    auto fs = stagger(grid.shape, d);
    auto hi = i;
    hi[d]++;
    const auto *f = g.face_fraction[d].const_data_ptr<double>();
    w[d] = h[(d + 1) % 3] * h[(d + 2) % 3] *
           (f[offset(i, fs)] - f[offset(hi, fs)]);
  }
  return w;
}

}  // namespace

EBGeometry EBGeometry::from_level_set(const Tensor &phi, const Tensor &x1f,
                                      const Tensor &x2f, const Tensor &x3f) {
  Grid grid(x1f, x2f, x3f);
  const auto nodes = host(phi);
  auto ns = grid.shape;
  for (auto &n : ns) ++n;
  TORCH_CHECK(nodes.sizes() == dims(ns),
              "nodal EB field must have shape [nz+1,ny+1,nx+1]");
  const auto *values = nodes.const_data_ptr<double>();
  auto g = allocate(grid.shape);
  for (int d = 0; d < 3; ++d) {
    auto fs = stagger(grid.shape, d);
    int u = (d + 1) % 3, v = (d + 2) % 3;
    each(fs, [&](Index i) {
      const Vec origin = grid.lower(i);
      std::array<Vertex, 4> corners;
      for (int b = 0; b < 4; ++b) {
        auto j = i;
        j[u] += b & 1;
        j[v] += b >> 1;
        corners[b] = {grid.lower(j) - origin, values[offset(j, ns)]};
        TORCH_CHECK(std::isfinite(corners[b].phi), "non-finite EB level set");
      }
      TORCH_CHECK(!(corners[0].phi > 0 && corners[3].phi > 0 &&
                    corners[1].phi < 0 && corners[2].phi < 0) &&
                      !(corners[0].phi < 0 && corners[3].phi < 0 &&
                        corners[1].phi > 0 && corners[2].phi > 0),
                  "multivalued EB on Cartesian face");
      Moments face;
      clip_triangle({corners[0], corners[1], corners[3]}, face);
      clip_triangle({corners[0], corners[2], corners[3]}, face);
      const auto t = offset(i, fs);
      const auto diagonal = corners[3].p;
      g.face_fraction[d].data_ptr<double>()[t] =
          std::clamp(face.measure / (diagonal[u] * diagonal[v]), 0.0, 1.0);
      write_vec(g.face_centroid[d], t, origin + face.centroid(0.5 * diagonal));
    });
  }
  each(grid.shape, [&](Index i) {
    auto t = offset(i, grid.shape);
    const auto origin = grid.lower(i), h = grid.width(i);
    std::array<Vertex, 8> c;
    int positive = 0, negative = 0;
    for (int b = 0; b < 8; ++b) {
      auto j = i;
      for (int d = 0; d < 3; ++d) j[d] += (b >> d) & 1;
      c[b] = {grid.lower(j) - origin, values[offset(j, ns)]};
      positive += c[b].phi > 0;
      negative += c[b].phi < 0;
    }
    check_topology(c);
    Moments volume, wall;
    // Six simplices defined by the six monotone corner-to-corner paths.
    for (int a = 0; a < 3; ++a)
      for (int b = 0; b < 3; ++b)
        if (b != a)
          clip_tet({c[0], c[1 << a], c[(1 << a) | (1 << b)], c[7]}, volume,
                   wall);
    double vf = std::clamp(volume.measure / (h[0] * h[1] * h[2]), 0.0, 1.0);
    if (!negative && positive &&
        std::none_of(c.begin(), c.end(),
                     [](const Vertex &p) { return p.phi == 0; }))
      vf = 1;
    // Round exactly full fluid volumes with aligned walls when every simplex
    // contains fluid and the only non-fluid vertices are on the isosurface.
    if (!negative && volume.measure > 0 &&
        std::abs(1 - vf) <= 8 * std::numeric_limits<double>::epsilon())
      vf = 1;
    g.volume_fraction.data_ptr<double>()[t] = vf;
    const auto center = 0.5 * h;
    write_vec(g.cell_centroid, t, origin + volume.centroid(center));
    write_vec(g.wall_centroid, t, origin + wall.centroid(center));
    auto w = wall_vector(g, grid, i, h);
    write_vec(g.wall_area_vector, t, w);
    g.flags.data_ptr<uint8_t>()[t] =
        vf == 0 ? 2 : (vf == 1 && wall.measure == 0 ? 0 : 1);
    TORCH_CHECK(wall.measure == 0 || std::hypot(w[0], w[1], w[2]) > 0,
                "embedded wall has degenerate net normal");
  });
  return g.to(phi.device());
}

EBGeometry EBGeometry::from_metrics(const Tensor &vf, const Tensor &cc,
                                    const std::array<Tensor, 3> &ff,
                                    const std::array<Tensor, 3> &fc,
                                    const Tensor &wv, const Tensor &wc,
                                    const Tensor &x1f, const Tensor &x2f,
                                    const Tensor &x3f) {
  Grid grid(x1f, x2f, x3f);
  double eps = 0;
  auto checked = [&](const Tensor &a, const Index &shape, bool vector) {
    auto h = host(a);
    TORCH_CHECK(h.sizes() == dims(shape, vector),
                "incorrect Cartesian EB metric shape");
    TORCH_CHECK(a.scalar_type() == torch::kFloat32 ||
                    a.scalar_type() == torch::kFloat64,
                "Cartesian EB metrics must be float32 or float64");
    eps = std::max(
        eps, a.scalar_type() == torch::kFloat64
                 ? std::numeric_limits<double>::epsilon()
                 : static_cast<double>(std::numeric_limits<float>::epsilon()));
    TORCH_CHECK(h.isfinite().all().item<bool>(),
                "non-finite Cartesian EB metric");
    if (!vector)
      TORCH_CHECK(((h >= 0) & (h <= 1)).all().item<bool>(),
                  "EB fraction outside [0,1]");
    return h.clone();
  };
  EBGeometry g;
  g.volume_fraction = checked(vf, grid.shape, false);
  g.cell_centroid = checked(cc, grid.shape, true);
  g.wall_area_vector = checked(wv, grid.shape, true);
  g.wall_centroid = checked(wc, grid.shape, true);
  g.flags = torch::empty(dims(grid.shape), torch::kUInt8);
  for (int d = 0; d < 3; ++d) {
    auto fs = stagger(grid.shape, d);
    g.face_fraction[d] = checked(ff[d], fs, false);
    g.face_centroid[d] = checked(fc[d], fs, true);
  }
  auto check_point = [&](const Vec &p, Vec lo, Vec hi) {
    for (int d = 0; d < 3; ++d) {
      double tol = 8 * eps * std::max(std::abs(lo[d]), std::abs(hi[d]));
      TORCH_CHECK(p[d] >= lo[d] - tol && p[d] <= hi[d] + tol,
                  "EB centroid outside its face or cell");
    }
  };
  for (int d = 0; d < 3; ++d) {
    auto fs = stagger(grid.shape, d);
    each(fs, [&](Index i) {
      auto t = offset(i, fs);
      if (g.face_fraction[d].const_data_ptr<double>()[t] == 0) return;
      auto hi = i;
      for (int e = 0; e < 3; ++e) hi[e] += e != d;
      check_point(read_vec(g.face_centroid[d], t), grid.lower(i),
                  grid.lower(hi));
    });
  }
  each(grid.shape, [&](Index i) {
    const auto t = offset(i, grid.shape);
    const auto v = g.volume_fraction.const_data_ptr<double>()[t];
    const auto h = grid.width(i), lo = grid.lower(i), hi = lo + h;
    auto w = read_vec(g.wall_area_vector, t),
         expected = wall_vector(g, grid, i, h);
    bool full_faces = true;
    for (int d = 0; d < 3; ++d) {
      auto fs = stagger(grid.shape, d), right = i;
      right[d]++;
      auto *f = g.face_fraction[d].const_data_ptr<double>();
      double l = f[offset(i, fs)], r = f[offset(right, fs)];
      TORCH_CHECK(v > 0 || (l == 0 && r == 0),
                  "open face incident on a covered EB cell");
      full_faces = full_faces && l == 1 && r == 1;
      const double area = h[(d + 1) % 3] * h[(d + 2) % 3];
      TORCH_CHECK(std::abs(w[d] - expected[d]) <= 64 * eps * area,
                  "embedded wall area vector violates Cartesian face closure");
    }
    const uint8_t flag = v == 0 ? 2 : (v == 1 && full_faces ? 0 : 1);
    g.flags.data_ptr<uint8_t>()[t] = flag;
    if (v > 0) check_point(read_vec(g.cell_centroid, t), lo, hi);
    if (flag == 1) {
      TORCH_CHECK(std::hypot(w[0], w[1], w[2]) > 0,
                  "cut cell has no net embedded wall normal");
      check_point(read_vec(g.wall_centroid, t), lo, hi);
    } else {
      TORCH_CHECK(w[0] == 0 && w[1] == 0 && w[2] == 0,
                  "regular or covered EB cell has a wall area vector");
    }
  });
  return g.to(vf.device());
}

EBGeometry EBGeometry::to(torch::Device device) const {
  // MPS implements float32 metrics. Host and CUDA retain geometry precision.
  auto move = [&](const Tensor &t) {
    return t.to(device, device.is_mps() ? torch::kFloat32 : torch::kFloat64);
  };
  EBGeometry g;
  g.volume_fraction = move(volume_fraction);
  g.cell_centroid = move(cell_centroid);
  g.wall_area_vector = move(wall_area_vector);
  g.wall_centroid = move(wall_centroid);
  g.flags = flags.to(device);
  for (int d = 0; d < 3; ++d) {
    g.face_fraction[d] = move(face_fraction[d]);
    g.face_centroid[d] = move(face_centroid[d]);
  }
  return g;
}

}  // namespace snap
