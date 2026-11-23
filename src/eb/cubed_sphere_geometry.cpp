#include <snap/layout/cubed_sphere_constants.h>

#include <limits>

#include "geometry_internal.hpp"

namespace snap {
namespace {
using namespace eb_detail;

double norm(const Vec& x) { return std::hypot(x[0], x[1], x[2]); }

// q=(r,tan(alpha),tan(beta)). Both angular mesh lines are great circles.
struct Chart {
  Vec pole, u, v;
  explicit Chart(int panel) {
    TORCH_CHECK(panel >= 0 && panel < 6, "cubed-sphere panel must be in [0,5]");
    auto direction = [&](double a, double b) {
      Vec p;
      cs_face_xyz_from_tan(panel, a, b, &p[0], &p[1], &p[2]);
      return p;
    };
    pole = direction(0, 0);
    u = direction(1, 0) - pole;
    v = direction(0, 1) - pole;
  }
  double length(const Vec& q) const { return std::hypot(1., q[1], q[2]); }
  Vec normal(const Vec& q) const {
    return (1 / length(q)) * (pole + q[1] * u + q[2] * v);
  }
  // The physical derivative of r*n in an arbitrary logical direction e.
  Vec derivative(const Vec& q, const Vec& n, const Vec& e) const {
    Vec tangent = e[1] * u + e[2] * v;
    return e[0] * n + (q[0] / length(q)) * (tangent - dot(tangent, n) * n);
  }
};

Grid chart_grid(const Tensor& r, const Tensor& a, const Tensor& b) {
  Grid g(r, a, b);
  TORCH_CHECK(g.axis[0][0].item<double>() > 0,
              "cubed-sphere EB requires positive radii, including ghost faces");
  for (int d = 1; d < 3; ++d) {
    TORCH_CHECK(g.axis[d].abs().max().item<double>() < std::acos(-1.) / 2,
                "cubed-sphere EB angular faces, including ghosts, must be "
                "inside the gnomonic chart");
    g.axis[d] = g.axis[d].tan();
  }
  return g;
}

double angle(const Vec& a, const Vec& b) {
  return std::atan2(norm(cross(a, b)), dot(a, b));
}
Vec arc_correction(const Vec& a, const Vec& b) {
  const Vec chord = cross(a, b - a);
  double sine = norm(chord);
  double theta = std::atan2(sine, dot(a, b));
  if (theta < 0.01) {
    // theta/sin(theta) - 1 with no loss for a nearly closed aperture.
    double t = theta * theta;
    return (0.5 * t * (1. / 6 + t * (7. / 360 + 31. * t / 15120))) * chord;
  }
  return (0.5 * (theta / sine - 1)) * chord;
}
struct AngularMoments {
  double omega;
  Vec normal_integral;
};
AngularMoments triangle_angles(const Vec& a, const Vec& b, const Vec& c) {
  // Solid angle of a spherical triangle and 1/2 integral n x dn along its
  // three great-circle edges (Stokes' theorem).
  double triple = dot(a, cross(b - a, c - a));
  return {
      2 * std::atan2(std::abs(triple), 1 + dot(a, b) + dot(b, c) + dot(c, a)),
      0.5 * cross(b - a, c - a) + arc_correction(a, b) + arc_correction(b, c) +
          arc_correction(c, a)};
}
AngularMoments angular_cell(const Chart& map, const Vec& lo, const Vec& hi) {
  auto a = map.normal(lo), b = map.normal({0, hi[1], lo[2]});
  auto c = map.normal(hi), d = map.normal({0, lo[1], hi[2]});
  auto one = triangle_angles(a, b, c), two = triangle_angles(a, c, d);
  return {one.omega + two.omega, one.normal_integral + two.normal_integral};
}

struct Measure {
  double amount;
  Vec center;
  Vec area_vector{};
};
Measure shell(const AngularMoments& angle, double lo, double hi) {
  double r2 = lo * lo + lo * hi + hi * hi;
  double mean_r = 0.75 * (hi + lo) * (hi * hi + lo * lo) / r2;
  return {(hi - lo) * r2 / 3 * angle.omega,
          (mean_r / angle.omega) * angle.normal_integral};
}

// Exact metrics of an uncut coordinate rectangle. Lateral faces lie in
// planes through the sphere center. Their along-edge direction traces an arc.
Measure rectangle(const Chart& map, int dir, const Vec& lo, const Vec& hi) {
  if (dir == 0) {
    auto a = angular_cell(map, lo, hi);
    double r = lo[0];
    return {r * r * a.omega, (r / a.omega) * a.normal_integral,
            r * r * a.normal_integral};
  }
  auto a = map.normal(lo), b = map.normal(hi);
  auto theta = angle(a, b);
  double area = 0.5 * (hi[0] - lo[0]) * (hi[0] + lo[0]) * theta;
  double mean_r = (2. / 3) * (hi[0] * hi[0] + hi[0] * lo[0] + lo[0] * lo[0]) /
                  (hi[0] + lo[0]);
  Vec normal = dir == 1 ? map.u - lo[1] * map.pole : map.v - lo[2] * map.pole;
  return {area, (mean_r * std::tan(theta / 2) / theta) * (a + b),
          (area / norm(normal)) * normal};
}

// Six-point Gauss-Legendre rule on [0,1]. Duffy coordinates turn each
// clipped simplex into a product rule. Only mixed cells use volume quadrature.
constexpr double point[] = {0.033765242898424, 0.169395306766868,
                            0.380690406958402, 0.619309593041598,
                            0.830604693233132, 0.966234757101576};
constexpr double weight[] = {0.085662246189585, 0.180380786524069,
                             0.233956967286346, 0.233956967286346,
                             0.180380786524069, 0.085662246189585};

struct Volume : Moments {
  const Chart& map;
  explicit Volume(const Chart& c) : map(c) {}
  void tet(const Vec& a, const Vec& b, const Vec& c, const Vec& d) {
    const Vec e = b - a, f = c - a, h = d - a;
    const double det = std::abs(dot(e, cross(f, h)));
    if (!det) return;
    for (int i = 0; i < 6; ++i)
      for (int j = 0; j < 6; ++j) {
        double s = 1 - point[i], t = s * (1 - point[j]);
        auto q0 = a + point[i] * e + (s * point[j]) * f;
        auto area = det * weight[i] * weight[j] * s * t;
        for (int k = 0; k < 6; ++k) {
          auto q = q0 + (t * point[k]) * h;
          double l = map.length(q), jac = q[0] * q[0] / (l * l * l);
          add(area * weight[k] * jac, q[0] * map.normal(q));
        }
      }
  }
};
struct Surface : Moments {
  const Chart& map;
  int dir;  // -1 is the general wall surface.
  Vec area_vector{};
  Surface(const Chart& c, int d) : map(c), dir(d) {}
  void triangle(const Vec& a, const Vec& b, const Vec& c) {
    const Vec e = b - a, f = c - a;
    if (norm(cross(e, f)) == 0) return;
    if (dir == 0) {
      auto t = triangle_angles(map.normal(a), map.normal(b), map.normal(c));
      if (t.omega == 0) return;
      auto av = (a[0] * a[0]) * t.normal_integral;
      area_vector = area_vector + av;
      add(a[0] * a[0] * t.omega, (a[0] / t.omega) * t.normal_integral);
      return;
    }
    for (int i = 0; i < 6; ++i)
      for (int j = 0; j < 6; ++j) {
        double t = 1 - point[i];
        auto q = a + point[i] * e + (t * point[j]) * f;
        auto n = map.normal(q);
        auto J = cross(map.derivative(q, n, e), map.derivative(q, n, f));
        double w = weight[i] * weight[j] * t;
        add(w * norm(J), q[0] * n);
        area_vector = area_vector + w * J;
      }
  }
};

template <size_t N>
Vertex middle(const std::array<Vertex, N>& vertices) {
  Vertex c{};
  for (auto const& p : vertices) {
    c.p = c.p + (1. / N) * p.p;
    c.phi += p.phi / N;
  }
  return c;
}
constexpr int cycle[] = {0, 1, 3, 2};
std::array<Vertex, 4> quad(const std::array<Vertex, 8>& corners, int d,
                           int side) {
  std::array<Vertex, 4> q;
  for (int j = 0; j < 4; ++j)
    q[j] = corners[(side << d) | ((j & 1) << ((d + 1) % 3)) |
                   ((j >> 1) << ((d + 2) % 3))];
  return q;
}

struct Mesh {
  const Chart map;
  const Grid grid;
  Tensor nodes;
  Index ns;
  Mesh(const Tensor& phi, const Tensor& r, const Tensor& a, const Tensor& b,
       int panel)
      : map(panel),
        grid(chart_grid(r, a, b)),
        nodes(host(phi)),
        ns(grid.shape) {
    for (auto& n : ns) ++n;
    TORCH_CHECK(
        nodes.sizes() == dims(ns) && nodes.isfinite().all().item<bool>(),
        "cubed-sphere EB needs finite nodal phi[nz+1,ny+1,nx+1]");
  }
  Vertex node(const Index& i) const {
    return {grid.lower(i), nodes.const_data_ptr<double>()[offset(i, ns)]};
  }

  void face(EBGeometry& g, int d, const Index& i) const {
    Index hi = i;
    for (int e = 0; e < 3; ++e) hi[e] += e != d;
    auto lo = grid.lower(i), up = grid.lower(hi);
    const auto full = rectangle(map, d, lo, up);
    std::array<Vertex, 4> q;
    for (int j = 0; j < 4; ++j) {
      auto p = i;
      p[(d + 1) % 3] += j & 1;
      p[(d + 2) % 3] += j >> 1;
      q[j] = node(p);
    }
    bool positive = false, negative = false;
    for (const auto& v : q) {
      positive |= v.phi > 0;
      negative |= v.phi < 0;
    }
    TORCH_CHECK(
        !(q[0].phi > 0 && q[3].phi > 0 && q[1].phi < 0 && q[2].phi < 0) &&
            !(q[0].phi < 0 && q[3].phi < 0 && q[1].phi > 0 && q[2].phi > 0),
        "multivalued cubed-sphere EB on a face");
    Measure fluid = full;
    if (!positive)
      fluid = {0, full.center, {}};
    else if (negative) {
      // A cut along one chart coordinate leaves an exact coordinate rectangle.
      int bit = q[0].phi == q[2].phi && q[1].phi == q[3].phi
                    ? 1
                    : (q[0].phi == q[1].phi && q[2].phi == q[3].phi ? 2 : 0);
      if (bit) {
        int e = (d + bit) % 3;
        double cut = crossing(q[0], q[bit]).at(e);
        if (q[0].phi > 0)
          up[e] = cut;
        else
          lo[e] = cut;
        fluid = rectangle(map, d, lo, up);
      } else {
        Surface surface(map, d);
        auto center = middle(q);
        for (int e = 0; e < 4; ++e)
          clip_triangle({center, q[cycle[e]], q[cycle[(e + 1) % 4]]}, surface);
        fluid = {surface.measure, surface.centroid(full.center),
                 surface.area_vector};
      }
    }
    auto j = offset(i, stagger(grid.shape, d));
    g.background_face_area[d].data_ptr<double>()[j] = full.amount;
    g.face_fraction[d].data_ptr<double>()[j] =
        std::clamp(fluid.amount / full.amount, 0., 1.);
    write_vec(g.face_centroid[d], j, fluid.center);
    write_vec(g.face_area_vector[d], j, fluid.area_vector);
  }

  void cell(EBGeometry& g, const Index& i) const {
    const auto lo = grid.lower(i),
               hi = grid.lower({i[0] + 1, i[1] + 1, i[2] + 1});
    const auto angle = angular_cell(map, lo, hi);
    const auto full = shell(angle, lo[0], hi[0]);
    std::array<Vertex, 8> c;
    int npos = 0, nneg = 0;
    for (int j = 0; j < 8; ++j) {
      Index p = i;
      for (int d = 0; d < 3; ++d) p[d] += (j >> d) & 1;
      c[j] = node(p);
      npos += c[j].phi > 0;
      nneg += c[j].phi < 0;
    }
    check_topology(c);
    Vec wall{};
    int closed = 0;
    for (int d = 0; d < 3; ++d) {
      auto upper = i;
      ++upper[d];
      auto fs = stagger(grid.shape, d);
      auto l = offset(i, fs), r = offset(upper, fs);
      wall = wall + (read_vec(g.face_area_vector[d], l) -
                     read_vec(g.face_area_vector[d], r));
      closed += g.face_fraction[d].const_data_ptr<double>()[l] == 0;
      closed += g.face_fraction[d].const_data_ptr<double>()[r] == 0;
    }
    const int flag = !npos ? 2 : (nneg || closed ? 1 : 0);
    auto fluid = full;
    Vec wall_center = full.center;
    if (!npos)
      fluid.amount = 0;
    else if (flag == 1) {
      bool radial = true;
      for (int j = 2; j < 8; ++j) radial &= c[j].phi == c[j & 1].phi;
      if (radial && c[0].phi != c[1].phi) {
        auto rc = crossing(c[0], c[1]).at(0);
        fluid =
            c[0].phi > 0 ? shell(angle, lo[0], rc) : shell(angle, rc, hi[0]);
        wall_center = (rc / angle.omega) * angle.normal_integral;
      } else {
        Volume vol(map);
        Surface surface(map, -1);
        const bool complement = nneg > 0 && npos > nneg;
        if (complement)
          for (auto& v : c) v.phi = -v.phi;
        auto center = middle(c);
        for (int d = 0; d < 3; ++d)
          for (int side = 0; side < 2; ++side) {
            auto q = quad(c, d, side);
            auto mid = middle(q);
            for (int e = 0; e < 4; ++e)
              clip_tet({center, mid, q[cycle[e]], q[cycle[(e + 1) % 4]]}, vol,
                       surface);
          }
        if (nneg) {
          fluid.amount = complement ? full.amount - vol.measure : vol.measure;
          fluid.center =
              complement
                  ? (1 / fluid.amount) * (full.amount * full.center - vol.first)
                  : vol.centroid(full.center);
        }
        wall_center = surface.centroid(full.center);
      }
    }
    auto j = offset(i, grid.shape);
    g.background_volume.data_ptr<double>()[j] = full.amount;
    g.volume_fraction.data_ptr<double>()[j] =
        std::clamp(fluid.amount / full.amount, 0., 1.);
    g.flags.data_ptr<uint8_t>()[j] = flag;
    write_vec(g.cell_centroid, j, fluid.center);
    write_vec(g.wall_centroid, j, wall_center);
    write_vec(g.wall_area_vector, j, flag == 1 ? wall : Vec{});
    TORCH_CHECK(flag != 1 || norm(wall) > 0,
                "cubed-sphere cut cell has a degenerate net normal");
  }
};

}  // namespace

EBGeometry EBGeometry::from_cubed_sphere_level_set(const Tensor& phi,
                                                   const Tensor& x1f,
                                                   const Tensor& x2f,
                                                   const Tensor& x3f,
                                                   int panel) {
  Mesh mesh(phi, x1f, x2f, x3f, panel);
  const auto shape = mesh.grid.shape;
  auto g = allocate(shape);
  g.panel = panel;
  g.background_volume = torch::empty_like(g.volume_fraction);
  for (int d = 0; d < 3; ++d) {
    g.background_face_area[d] = torch::empty_like(g.face_fraction[d]);
    g.face_area_vector[d] = torch::empty_like(g.face_centroid[d]);
    each(stagger(shape, d), [&](Index i) { mesh.face(g, d, i); });
  }
  each(shape, [&](Index i) { mesh.cell(g, i); });
  return g.to(phi.device());
}

EBGeometry EBGeometry::from_cubed_sphere_metrics(
    const Tensor& vf, const Tensor& cc, const std::array<Tensor, 3>& ff,
    const std::array<Tensor, 3>& fc, const std::array<Tensor, 3>& av,
    const Tensor& wv, const Tensor& wc, const Tensor& r, const Tensor& a,
    const Tensor& b, int panel) {
  auto grid = chart_grid(r, a, b);
  Chart map(panel);
  // The analytic full-cell metrics form the independent reference for units,
  // orientation, and shared face support of user-supplied cut metrics.
  auto nodes = torch::ones({b.numel(), a.numel(), r.numel()}, torch::kFloat64);
  auto g = from_cubed_sphere_level_set(nodes, r, a, b, panel);
  double eps = std::numeric_limits<double>::epsilon();
  auto checked = [&](const Tensor& t, const Index& shape, bool vector) {
    auto h = host(t);
    TORCH_CHECK(t.scalar_type() == torch::kFloat32 ||
                    t.scalar_type() == torch::kFloat64,
                "cubed-sphere EB metrics must use float32 or float64");
    if (t.scalar_type() == torch::kFloat32)
      eps = std::numeric_limits<float>::epsilon();
    TORCH_CHECK(
        h.sizes() == dims(shape, vector) && h.isfinite().all().item<bool>(),
        "invalid shape or nonfinite cubed-sphere EB metric");
    if (!vector)
      TORCH_CHECK(((h >= 0) & (h <= 1)).all().item<bool>(),
                  "cubed-sphere EB fraction outside [0,1]");
    return h.clone();
  };
  auto full_face = g.face_area_vector;
  g.volume_fraction = checked(vf, grid.shape, false);
  g.cell_centroid = checked(cc, grid.shape, true);
  g.wall_area_vector = checked(wv, grid.shape, true);
  g.wall_centroid = checked(wc, grid.shape, true);
  for (int d = 0; d < 3; ++d) {
    auto fs = stagger(grid.shape, d);
    g.face_fraction[d] = checked(ff[d], fs, false);
    g.face_centroid[d] = checked(fc[d], fs, true);
    g.face_area_vector[d] = checked(av[d], fs, true);
  }
  // Spherical face/cell averages are in the convex hull of the physical
  // wedge; their radius can be smaller than the shell's inner radius.
  auto check_point = [&](const Vec& p, const Vec& lo, const Vec& hi) {
    double scale = hi[0], tol = 256 * eps * scale, pole = dot(p, map.pole);
    double u = dot(p, map.u), v = dot(p, map.v);
    TORCH_CHECK(norm(p) <= hi[0] + tol && pole > 0 && u >= lo[1] * pole - tol &&
                    u <= hi[1] * pole + tol && v >= lo[2] * pole - tol &&
                    v <= hi[2] * pole + tol,
                "cubed-sphere EB centroid outside its physical patch");
    double corner_length =
        std::hypot(1., std::max(std::abs(lo[1]), std::abs(hi[1])),
                   std::max(std::abs(lo[2]), std::abs(hi[2])));
    TORCH_CHECK(pole + tol >= lo[0] / corner_length,
                "cubed-sphere EB centroid below shell");
  };
  for (int d = 0; d < 3; ++d) {
    auto fs = stagger(grid.shape, d);
    each(fs, [&](Index i) {
      auto j = offset(i, fs);
      auto f = g.face_fraction[d].const_data_ptr<double>()[j];
      auto vec = read_vec(g.face_area_vector[d], j),
           unclipped = read_vec(full_face[d], j);
      const double full = g.background_face_area[d].const_data_ptr<double>()[j],
                   area = f * full;
      if (f == 0) {
        TORCH_CHECK(norm(vec) == 0, "closed EB face has nonzero vector area");
        return;
      }
      TORCH_CHECK(norm(vec) > 0 && dot(vec, unclipped) > 0 &&
                      norm(vec) <= area * (1 + 256 * eps),
                  "invalid oriented cubed-sphere face area");
      auto h = i;
      for (int e = 0; e < 3; ++e) h[e] += e != d;
      auto lo = grid.lower(i), hi = grid.lower(h);
      auto center = read_vec(g.face_centroid[d], j);
      check_point(center, lo, hi);
      auto expected = d == 0 ? (area / lo[0]) * center : f * unclipped;
      TORCH_CHECK(norm(vec - expected) <= 256 * eps * area,
                  "cubed-sphere EB face vector inconsistent with face metric");
    });
  }
  each(grid.shape, [&](Index i) {
    auto j = offset(i, grid.shape);
    double fluid = g.volume_fraction.const_data_ptr<double>()[j];
    auto w = read_vec(g.wall_area_vector, j);
    Vec closure = w;
    bool full = true;
    double area = 0;
    for (int d = 0; d < 3; ++d) {
      auto fs = stagger(grid.shape, d), hi = i;
      ++hi[d];
      auto l = offset(i, fs), r = offset(hi, fs);
      auto* face = g.face_fraction[d].const_data_ptr<double>();
      full &= face[l] == 1 && face[r] == 1;
      TORCH_CHECK(fluid > 0 || (face[l] == 0 && face[r] == 0),
                  "open cubed-sphere face touches covered cell");
      auto left = read_vec(g.face_area_vector[d], l),
           right = read_vec(g.face_area_vector[d], r);
      closure = closure + (right - left);
      area += norm(left) + norm(right);
    }
    TORCH_CHECK(norm(closure) <= 512 * eps * area,
                "cubed-sphere EB violates physical area-vector closure");
    int flag = fluid == 0 ? 2 : (fluid == 1 && full ? 0 : 1);
    TORCH_CHECK((flag == 1) == (norm(w) > 0),
                "inconsistent cubed-sphere wall area and cell type");
    g.flags.data_ptr<uint8_t>()[j] = flag;
    auto hi = i;
    for (auto& x : hi) ++x;
    if (fluid > 0)
      check_point(read_vec(g.cell_centroid, j), grid.lower(i), grid.lower(hi));
    if (flag == 1)
      check_point(read_vec(g.wall_centroid, j), grid.lower(i), grid.lower(hi));
  });
  return g.to(vf.device());
}

}  // namespace snap
