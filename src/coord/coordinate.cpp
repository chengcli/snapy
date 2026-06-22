// C/C++
#include <vector>

// yaml
#include <yaml-cpp/yaml.h>

// snap
#include <snap/snap.h>

#include <snap/layout/layout.hpp>
#include <snap/mesh/meshblock.hpp>

#include "coordinate.hpp"
#include "gnomonic_equiangle.hpp"
#include "spherical_polar.hpp"

namespace snap {

torch::Tensor derivative(torch::Tensor value, torch::Tensor distance, int dim) {
  auto grad = torch::zeros_like(value);
  int n = value.size(dim);
  if (n <= 1) return grad;

  auto full = torch::indexing::Slice();
  std::vector<torch::indexing::TensorIndex> dst(value.dim(), full);
  std::vector<torch::indexing::TensorIndex> lo(value.dim(), full);
  std::vector<torch::indexing::TensorIndex> hi(value.dim(), full);

  dst[dim] = 0;
  lo[dim] = 0;
  hi[dim] = 1;
  grad.index_put_(dst, (value.index(hi) - value.index(lo)) /
                           distance.index(hi).clamp_min(1.e-30));

  if (n > 2) {
    dst[dim] = torch::indexing::Slice(1, n - 1);
    lo[dim] = torch::indexing::Slice(0, n - 2);
    hi[dim] = torch::indexing::Slice(2, n);

    std::vector<torch::indexing::TensorIndex> d_lo(distance.dim(), full);
    std::vector<torch::indexing::TensorIndex> d_hi(distance.dim(), full);
    d_lo[dim] = torch::indexing::Slice(1, n - 1);
    d_hi[dim] = torch::indexing::Slice(2, n);
    auto denom = distance.index(d_lo) + distance.index(d_hi);
    grad.index_put_(
        dst, (value.index(hi) - value.index(lo)) / denom.clamp_min(1.e-30));
  }

  dst[dim] = n - 1;
  lo[dim] = n - 2;
  hi[dim] = n - 1;
  grad.index_put_(dst, (value.index(hi) - value.index(lo)) /
                           distance.index(hi).clamp_min(1.e-30));

  return grad;
}

CoordinateOptions CoordinateOptionsImpl::from_yaml(
    std::string const& filename) {
  auto op = CoordinateOptionsImpl::create();

  auto config = YAML::LoadFile(filename);
  auto node = config["geometry"];
  if (!node) return op;  // return default options

  op->type(node["type"].as<std::string>("cartesian"));

  double x1min = 0, x2min = 0, x3min = 0, x1max = 1, x2max = 1, x3max = 1;

  if (node["bounds"]) {
    x1min = node["bounds"]["x1min"].as<double>(0.0);

    if (node["bounds"]["x2min_pi"]) {
      x2min = node["bounds"]["x2min_pi"].as<double>() * M_PI;
    } else {
      x2min = node["bounds"]["x2min"].as<double>(0.0);
    }

    if (node["bounds"]["x3min_pi"]) {
      x3min = node["bounds"]["x3min_pi"].as<double>() * M_PI;
    } else {
      x3min = node["bounds"]["x3min"].as<double>(0.0);
    }

    x1max = node["bounds"]["x1max"].as<double>(1.0);

    if (node["bounds"]["x2max_pi"]) {
      x2max = node["bounds"]["x2max_pi"].as<double>() * M_PI;
    } else {
      x2max = node["bounds"]["x2max"].as<double>(1.0);
    }

    if (node["bounds"]["x3max_pi"]) {
      x3max = node["bounds"]["x3max_pi"].as<double>() * M_PI;
    } else {
      x3max = node["bounds"]["x3max"].as<double>(1.0);
    }
  }

  op->global_x1min() = x1min;
  op->global_x2min() = x2min;
  op->global_x3min() = x3min;
  op->global_x1max() = x1max;
  op->global_x2max() = x2max;
  op->global_x3max() = x3max;

  if (!node["cells"]) return op;

  op->global_nx1() = node["cells"]["nx1"].as<int>(1);
  op->global_nx2() = node["cells"]["nx2"].as<int>(1);
  op->global_nx3() = node["cells"]["nx3"].as<int>(1);

  auto layout = LayoutOptionsImpl::from_yaml(filename);
  auto playout =
      LayoutImpl::create(std::make_shared<LayoutOptionsImpl>(*layout));
  int rank = playout->options->rank();
  auto [lx2, lx3, lx1] = playout->loc_of(rank);

  if (playout->options->type() == "cubed-sphere") lx1 = 0;

  int nb1 = playout->options->pz();
  int nb2 = playout->options->px();
  int nb3 = playout->options->py();

  if (op->global_nx1() % nb1 != 0) {
    TORCH_CHECK(
        false,
        "Number of total x1 grids must be divisible by the number of mesh "
        "blocks in x1 direction");
  }

  if (op->global_nx2() % nb2 != 0) {
    TORCH_CHECK(
        false,
        "Number of total x2 grids must be divisible by the number of mesh "
        "blocks in x2 direction");
  }

  if (op->global_nx3() % nb3 != 0) {
    TORCH_CHECK(
        false,
        "Number of totla x3 grids must be divisible by the number of mesh "
        "blocks in x3 direction");
  }

  op->nghost() = node["cells"]["nghost"].as<int>(1);

  if (op->nx1() > 1 && op->nx1() < op->nghost()) {
    TORCH_CHECK(false,
                "Number of x1 grids must be greater than the ghost zone size");
  }

  if (op->nx2() > 1 && op->nx2() < op->nghost()) {
    TORCH_CHECK(false,
                "Number of x2 grids must be greater than the ghost zone size");
  }

  if (op->nx3() > 1 && op->nx3() < op->nghost()) {
    TORCH_CHECK(false,
                "Number of x3 grids must be greater than the ghost zone size");
  }

  op->interp_order() = node["cells"]["interp_order"].as<int>(2);
  op->repartition(layout);
  return op;
}

void CoordinateOptionsImpl::repartition(LayoutOptions const& layout) {
  auto playout =
      LayoutImpl::create(std::make_shared<LayoutOptionsImpl>(*layout));
  int rank = layout->rank();
  auto [lx2, lx3, lx1] = playout->loc_of(rank);

  if (layout->type() == "cubed-sphere") lx1 = 0;

  int nb1 = layout->pz();
  int nb2 = layout->px();
  int nb3 = layout->py();

  x1min(global_x1min() + lx1 * (global_x1max() - global_x1min()) / nb1);
  x1max(x1min() + (global_x1max() - global_x1min()) / nb1);

  x2min(global_x2min() + lx2 * (global_x2max() - global_x2min()) / nb2);
  x2max(x2min() + (global_x2max() - global_x2min()) / nb2);

  x3min(global_x3min() + lx3 * (global_x3max() - global_x3min()) / nb3);
  x3max(x3min() + (global_x3max() - global_x3min()) / nb3);

  nx1(global_nx1() / nb1);
  nx2(global_nx2() / nb2);
  nx3(global_nx3() / nb3);
}

CoordinateImpl::CoordinateImpl(const CoordinateOptions& options_,
                               torch::nn::Module* p)
    : options(options_) {
  pmb = dynamic_cast<MeshBlockImpl const*>(p);

  auto const& op = options;

  auto dx = (op->x1max() - op->x1min()) / op->nx1();
  auto x1min = op->nx1() > 1 ? op->x1min() - op->nghost() * dx : op->x1min();
  auto x1max = op->nx1() > 1 ? op->x1max() + op->nghost() * dx : op->x1max();
  x1f = torch::linspace(x1min, x1max, op->nc1() + 1, torch::kFloat64);

  dx = (op->x2max() - op->x2min()) / op->nx2();
  auto x2min = op->nx2() > 1 ? op->x2min() - op->nghost() * dx : op->x2min();
  auto x2max = op->nx2() > 1 ? op->x2max() + op->nghost() * dx : op->x2max();
  x2f = torch::linspace(x2min, x2max, op->nc2() + 1, torch::kFloat64);

  dx = (op->x3max() - op->x3min()) / op->nx3();
  auto x3min = op->nx3() > 1 ? op->x3min() - op->nghost() * dx : op->x3min();
  auto x3max = op->nx3() > 1 ? op->x3max() + op->nghost() * dx : op->x3max();
  x3f = torch::linspace(x3min, x3max, op->nc3() + 1, torch::kFloat64);
}

void CoordinateImpl::reset_coordinates(std::array<MeshGenerator, 3> meshgens) {
  auto const& op = options;

  if (meshgens[0] != nullptr) {
    int nx1f = x1f.size(0);
    auto rx = compute_logical_position(
        torch::linspace(0, nx1f, nx1f, torch::kFloat64), nx1f, true);
    x1f.copy_(meshgens[0](rx, op->x1min(), op->x1max()));
  }

  if (meshgens[1] != nullptr) {
    int nx2f = x2f.size(0);
    auto rx = compute_logical_position(
        torch::linspace(0, nx2f, nx2f, torch::kFloat64), nx2f, true);
    x2f.copy_(meshgens[1](rx, op->x2min(), op->x2max()));
  }

  if (meshgens[2] != nullptr) {
    int nx3f = x3f.size(0);
    auto rx = compute_logical_position(
        torch::linspace(0, nx3f, nx3f, torch::kFloat64), nx3f, true);
    x3f.copy_(meshgens[2](rx, op->x3min(), op->x3max()));
  }
}

void CoordinateImpl::print(std::ostream& stream) const {
  stream << "x1f = [";
  for (int i = 0; i < x1f.size(0); ++i) {
    stream << x1f[i].item<float>();
    if (i < x1f.size(0) - 1) {
      stream << ", ";
    }
  }
  stream << "]" << std::endl << "x1v = [";
  for (int i = 0; i < x1v.size(0); ++i) {
    stream << x1v[i].item<float>();
    if (i < x1v.size(0) - 1) {
      stream << ", ";
    }
  }
  stream << "]" << std::endl;

  stream << "x2f = [";
  for (int i = 0; i < x2f.size(0); ++i) {
    stream << x2f[i].item<float>();
    if (i < x2f.size(0) - 1) {
      stream << ", ";
    }
  }
  stream << "]" << std::endl << "x2v = [";
  for (int i = 0; i < x2v.size(0); ++i) {
    stream << x2v[i].item<float>();
    if (i < x2v.size(0) - 1) {
      stream << ", ";
    }
  }
  stream << "]" << std::endl;

  stream << "x3f = [";
  for (int i = 0; i < x3f.size(0); ++i) {
    stream << x3f[i].item<float>();
    if (i < x3f.size(0) - 1) {
      stream << ", ";
    }
  }
  stream << "]" << std::endl << "x3v = [";
  for (int i = 0; i < x3v.size(0); ++i) {
    stream << x3v[i].item<float>();
    if (i < x3v.size(0) - 1) {
      stream << ", ";
    }
  }
  stream << "]" << std::endl;
}

torch::Tensor CoordinateImpl::center_width1() const {
  return dx1f.unsqueeze(0).unsqueeze(1);
}

torch::Tensor CoordinateImpl::center_width2() const {
  return dx2f.unsqueeze(0).unsqueeze(2);
}

torch::Tensor CoordinateImpl::center_width3() const {
  return dx3f.unsqueeze(1).unsqueeze(2);
}

torch::Tensor CoordinateImpl::center_distance1() const {
  return dx1v.unsqueeze(0).unsqueeze(1);
}

torch::Tensor CoordinateImpl::center_distance2() const {
  return dx2v.unsqueeze(0).unsqueeze(2);
}

torch::Tensor CoordinateImpl::center_distance3() const {
  return dx3v.unsqueeze(1).unsqueeze(2);
}

torch::Tensor CoordinateImpl::face_area1() const {
  return dx3f.outer(dx2f).unsqueeze(2).expand({-1, -1, x1f.size(0)});
}

torch::Tensor CoordinateImpl::face_area2() const {
  return dx3f.outer(dx1f).unsqueeze(1).expand({-1, x2f.size(0), -1});
}

torch::Tensor CoordinateImpl::face_area3() const {
  return dx2f.outer(dx1f).unsqueeze(0).expand({x3f.size(0), -1, -1});
}

torch::Tensor CoordinateImpl::cell_volume() const {
  return torch::einsum("km,mji->kji",
                       {dx3f.unsqueeze(1), dx2f.outer(dx1f).unsqueeze(0)});
}

torch::Tensor CoordinateImpl::find_cell_index(
    torch::Tensor const& coords) const {
  torch::Tensor index = torch::zeros_like(coords, torch::dtype(torch::kInt64));

  // x1dir
  index.slice(1, 0, 1) = torch::searchsorted(x1f, coords.slice(1, 0, 1));

  // x2dir
  if (coords.size(1) > 1) {
    index.slice(1, 1, 2) = torch::searchsorted(x2f, coords.slice(1, 1, 2));
  }

  // x3dir
  if (coords.size(1) > 2) {
    index.slice(1, 2, 3) = torch::searchsorted(x3f, coords.slice(1, 2, 3));
  }
  return index;
}

torch::Tensor CoordinateImpl::divergence(torch::Tensor flux1,
                                         torch::Tensor flux2,
                                         torch::Tensor flux3) const {
  enum { DIM1 = 3, DIM2 = 2, DIM3 = 1, DIMC = 0 };

  auto vol = cell_volume();

  torch::Tensor dflx;
  if (flux1.defined()) {
    dflx = torch::zeros_like(flux1);
  } else if (flux2.defined()) {
    dflx = torch::zeros_like(flux2);
  } else if (flux3.defined()) {
    dflx = torch::zeros_like(flux3);
  } else {
    TORCH_CHECK(false, "At least one flux tensor must be defined");
  }

  int si = il();
  int ei = iu() + 1;
  int sj = jl();
  int ej = ju() + 1;
  int sk = kl();
  int ek = ku() + 1;

  if (flux1.defined() > 0) {
    dflx.slice(DIM1, si, ei) +=
        face_area1(si + 1, ei + 1) * flux1.slice(DIM1, si + 1, ei + 1) -
        face_area1(si, ei) * flux1.slice(DIM1, si, ei);
  }

  if (flux2.defined() > 0) {
    dflx.slice(DIM2, sj, ej) +=
        face_area2(sj + 1, ej + 1) * flux2.slice(DIM2, sj + 1, ej + 1) -
        face_area2(sj, ej) * flux2.slice(DIM2, sj, ej);
  }

  if (flux3.defined() > 0) {
    dflx.slice(DIM3, sk, ek) +=
        face_area3(sk + 1, ek + 1) * flux3.slice(DIM3, sk + 1, ek + 1) -
        face_area3(sk, ek) * flux3.slice(DIM3, sk, ek);
  }

  return dflx / vol;
}

torch::Tensor CoordinateImpl::curl(torch::Tensor velocity) const {
  TORCH_CHECK(velocity.dim() == 4 && velocity.size(0) >= 3,
              "CoordinateImpl::curl expects a vector tensor with shape "
              "[3,nx3,nx2,nx1]");

  enum { DIM1 = 2, DIM2 = 1, DIM3 = 0 };

  auto v1 = velocity[VEL1];
  auto v2 = velocity[VEL2];
  auto v3 = velocity[VEL3];

  auto dtype = velocity.scalar_type();
  auto device = velocity.device();
  auto dx1 = center_distance1().to(device).to(dtype);
  auto dx2 = center_distance2().to(device).to(dtype);
  auto dx3 = center_distance3().to(device).to(dtype);
  auto dl1 = center_width1().to(device).to(dtype);
  auto dl2 = center_width2().to(device).to(dtype);
  auto dl3 = center_width3().to(device).to(dtype);

  auto curl1 = derivative(v3 * dl3, dx2, DIM2) / dl3.clamp_min(1.e-30) -
               derivative(v2 * dl2, dx3, DIM3) / dl2.clamp_min(1.e-30);
  auto curl2 = derivative(v1 * dl1, dx3, DIM3) / dl1.clamp_min(1.e-30) -
               derivative(v3 * dl3, dx1, DIM1) / dl3.clamp_min(1.e-30);
  auto curl3 = derivative(v2 * dl2, dx1, DIM1) / dl2.clamp_min(1.e-30) -
               derivative(v1 * dl1, dx2, DIM2) / dl1.clamp_min(1.e-30);

  return torch::stack({curl1, curl2, curl3});
}

torch::Tensor CoordinateImpl::forward(torch::Tensor prim, torch::Tensor flux1,
                                      torch::Tensor flux2,
                                      torch::Tensor flux3) {
  (void)prim;
  return divergence(flux1, flux2, flux3);
}

Coordinate CoordinateImpl::create(CoordinateOptions const& opts,
                                  torch::nn::Module* p,
                                  std::string const& name) {
  TORCH_CHECK(p, "[Coordinate] Parent module is null");
  TORCH_CHECK(opts, "[Coordinate] Options pointer is null");

  if (opts->type() == "cartesian") {
    return p->register_module(name, Cartesian(opts, p));
  } else if (opts->type() == "cylindrical") {
    return p->register_module(name, Cylindrical(opts, p));
  } else if (opts->type() == "spherical-polar") {
    return p->register_module(name, SphericalPolar(opts, p));
  } else if (opts->type() == "gnomonic-equiangle") {
    return p->register_module(name, GnomonicEquiangle(opts, p));
  } else {
    TORCH_CHECK(false, "Unknown coordinate type: ", opts->type());
  }
}
}  // namespace snap
