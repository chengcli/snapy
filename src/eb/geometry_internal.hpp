#pragma once

#include <algorithm>
#include <cmath>
#include <vector>

#include "geometry.hpp"

namespace snap::eb_detail {

using Tensor = torch::Tensor;
using Vec = std::array<double, 3>;
using Index = std::array<int64_t, 3>;

inline Vec operator+(Vec a, const Vec& b) {
  for (int d = 0; d < 3; ++d) a[d] += b[d];
  return a;
}
inline Vec operator-(Vec a, const Vec& b) {
  for (int d = 0; d < 3; ++d) a[d] -= b[d];
  return a;
}
inline Vec operator*(double s, Vec a) {
  for (auto& x : a) x *= s;
  return a;
}
inline Vec cross(const Vec& a, const Vec& b) {
  return {a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2],
          a[0] * b[1] - a[1] * b[0]};
}
inline double dot(const Vec& a, const Vec& b) {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

struct Moments {
  double measure{};
  Vec first{};
  void add(double w, const Vec& centroid) {
    measure += w;
    first = first + w * centroid;
  }
  Vec centroid(const Vec& fallback) const {
    if (measure == 0) return fallback;
    // Division componentwise also works for subnormal measures.
    return {first[0] / measure, first[1] / measure, first[2] / measure};
  }
  void triangle(const Vec& a, const Vec& b, const Vec& c) {
    const auto n = cross(b - a, c - a);
    add(std::hypot(n[0], n[1], n[2]) / 2, (1.0 / 3) * (a + b + c));
  }
  void tet(const Vec& a, const Vec& b, const Vec& c, const Vec& d) {
    add(std::abs(dot(b - a, cross(c - a, d - a))) / 6, 0.25 * (a + b + c + d));
  }
};

// Bottom abc and top ABC, with edges aA, bB, cC.
template <class Volume>
void prism(Volume& volume, const Vec& a, const Vec& b, const Vec& c,
           const Vec& A, const Vec& B, const Vec& C) {
  volume.tet(a, b, c, A);
  volume.tet(A, b, c, B);
  volume.tet(A, c, B, C);
}

struct Vertex {
  Vec p;
  double phi;
};
inline Vec crossing(const Vertex& a, const Vertex& b) {
  // Interpolate from the vertex nearest zero to resolve small fluid slivers.
  if (std::abs(a.phi) > std::abs(b.phi)) return crossing(b, a);
  const double ratio = std::abs(a.phi / b.phi);
  return a.p + (ratio / (1 + ratio)) * (b.p - a.p);
}

template <class Surface>
void clip_triangle(const std::array<Vertex, 3>& tri, Surface& face) {
  std::array<Vec, 4> polygon;
  int count = 0;
  for (int j = 0; j < 3; ++j) {
    const auto& a = tri[j];
    const auto& b = tri[(j + 1) % 3];
    if (a.phi > 0) polygon[count++] = a.p;
    if ((a.phi > 0) != (b.phi > 0)) polygon[count++] = crossing(a, b);
  }
  for (int j = 2; j < count; ++j)
    face.triangle(polygon[0], polygon[j - 1], polygon[j]);
}

template <class Volume, class Surface>
void clip_tet(const std::array<Vertex, 4>& tet, Volume& volume, Surface& wall) {
  std::array<Vertex, 4> in{}, out{};
  int n = 0, m = 0;
  for (const auto& v : tet) {
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
    prism(volume, in[0].p, in[1].p, in[2].p, p, q, r);
    wall.triangle(p, q, r);
    return;
  }
  auto p = crossing(in[0], out[0]), q = crossing(in[0], out[1]),
       r = crossing(in[1], out[0]), s = crossing(in[1], out[1]);
  prism(volume, in[0].p, p, q, in[1].p, r, s);
  wall.triangle(p, q, r);
  wall.triangle(q, r, s);
}

inline Tensor host(const Tensor& t) {
  TORCH_CHECK(t.defined() && t.is_floating_point(),
              "EB metrics need floating point tensors");
  return t.detach().to(torch::kCPU, torch::kFloat64).contiguous();
}

struct Grid {
  std::array<Tensor, 3> axis;
  Index shape;
  Grid(const Tensor& x, const Tensor& y, const Tensor& z)
      : axis{host(x), host(y), host(z)} {
    for (int d = 0; d < 3; ++d) {
      TORCH_CHECK(axis[d].dim() == 1 && axis[d].numel() >= 2,
                  "EB coordinates must be one-dimensional face vectors");
      shape[d] = axis[d].numel() - 1;
      const auto* f = axis[d].const_data_ptr<double>();
      TORCH_CHECK(std::isfinite(f[0]), "non-finite EB coordinate");
      for (int64_t j = 1; j <= shape[d]; ++j)
        TORCH_CHECK(std::isfinite(f[j]) && f[j] > f[j - 1],
                    "EB face coordinates must increase strictly");
    }
  }
  Vec lower(const Index& i) const {
    return {axis[0].const_data_ptr<double>()[i[0]],
            axis[1].const_data_ptr<double>()[i[1]],
            axis[2].const_data_ptr<double>()[i[2]]};
  }
  Vec width(const Index& i) const {
    return lower({i[0] + 1, i[1] + 1, i[2] + 1}) - lower(i);
  }
};

inline int64_t offset(const Index& i, const Index& shape) {
  return (i[2] * shape[1] + i[1]) * shape[0] + i[0];
}
inline std::vector<int64_t> dims(const Index& shape, bool vector = false) {
  std::vector<int64_t> result{shape[2], shape[1], shape[0]};
  if (vector) result.insert(result.begin(), 3);
  return result;
}
template <class F>
void each(const Index& shape, F f) {
  for (int64_t k = 0; k < shape[2]; ++k)
    for (int64_t j = 0; j < shape[1]; ++j)
      for (int64_t i = 0; i < shape[0]; ++i) f(Index{i, j, k});
}
inline void write_vec(Tensor& t, int64_t i, const Vec& v) {
  const int64_t stride = t.numel() / 3;
  for (int d = 0; d < 3; ++d) t.data_ptr<double>()[d * stride + i] = v[d];
}
inline Vec read_vec(const Tensor& t, int64_t i) {
  const auto* p = t.const_data_ptr<double>() + i;
  const int64_t n = t.numel() / 3;
  return {p[0], p[n], p[2 * n]};
}
inline Index stagger(Index shape, int d) {
  shape[d]++;
  return shape;
}

inline EBGeometry allocate(const Index& shape) {
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
inline void check_topology(const std::array<Vertex, 8>& v) {
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

}  // namespace snap::eb_detail
