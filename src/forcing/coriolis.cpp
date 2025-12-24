// yaml
#include <yaml-cpp/yaml.h>

// snap
#include <snap/snap.h>

#include <snap/coord/coordinate.hpp>
#include <snap/coord/cubed_sphere_utils.hpp>
#include <snap/hydro/hydro.hpp>
#include <snap/layout/cubed_sphere_layout.hpp>
#include <snap/mesh/meshblock.hpp>

#include "forcing.hpp"

namespace snap {

CoriolisOptions CoriolisOptionsImpl::from_yaml(YAML::Node const &forcing) {
  if (!forcing["coriolis"]) return nullptr;

  auto node = forcing["coriolis"];
  auto op = CoriolisOptionsImpl::create();

  op->type() = node["type"].as<std::string>("xyz");
  TORCH_CHECK(op->type() == "xyz" || op->type() == "123",
              "CoriolisOptions: unsupported type ", op->type());

  op->omega1() = node["omega1"].as<double>(0.);
  op->omega2() = node["omega2"].as<double>(0.);
  op->omega3() = node["omega3"].as<double>(0.);

  return op;
}

Coriolis123Impl::Coriolis123Impl(CoriolisOptions const &options_,
                                 torch::nn::Module *p)
    : options(options_) {
  phydro = dynamic_cast<HydroImpl const *>(p);
  reset();
}

void Coriolis123Impl::reset() {
  TORCH_CHECK(phydro, "[Coriolis123] Parent Hydro is null");
}

torch::Tensor Coriolis123Impl::forward(torch::Tensor du, torch::Tensor w,
                                       torch::Tensor temp, double dt) {
  if (options->omega1() != 0.0 || options->omega2() != 0.0 ||
      options->omega3() != 0.0) {
    auto m1 = w[IDN] * w[IVX];
    auto m2 = w[IDN] * w[IVY];
    auto m3 = w[IDN] * w[IVZ];
    du[IVX] += 2. * dt * (options->omega3() * m2 - options->omega2() * m3);
    du[IVY] += 2. * dt * (options->omega1() * m3 - options->omega3() * m1);

    if (w.size(1) > 1) {  // 3d
      du[IVZ] += 2. * dt * (options->omega2() * m1 - options->omega1() * m2);
    }
  }

  return du;
}

CoriolisXYZImpl::CoriolisXYZImpl(CoriolisOptions const &options_,
                                 torch::nn::Module *p)
    : options(options_) {
  phydro = dynamic_cast<HydroImpl const *>(p);
  reset();
}

void CoriolisXYZImpl::reset() {
  TORCH_CHECK(phydro, "[CoriolisXYZ] Parent Hydro is null");
  auto pcoord = phydro->pmb->pcoord;

  auto mesh = torch::meshgrid({pcoord->x3v, pcoord->x2v, pcoord->x1v}, "ij");

  auto omegaz = options->omega1();
  auto omegax = options->omega2();
  auto omegay = options->omega3();

  if (pcoord->options->type() == "cartesian") {
    omega1 = omegaz * ones_like(mesh[0]);
    omega2 = omegax * ones_like(mesh[0]);
    omega3 = omegay * ones_like(mesh[0]);
  } else if (pcoord->options->type() == "cylindrical") {
    auto theta = mesh[1];

    omega1 = theta.cos() * omegax + theta.sin() * omegay;
    omega2 = -theta.sin() * omegax + theta.cos() * omegay;
    omega3 = omegaz * ones_like(mesh[0]);
  } else if (pcoord->options->type() == "spherical-polar") {
    auto theta = mesh[1];
    auto phi = mesh[0];

    omega1 = theta.sin() * phi.cos() * omegax +
             theta.sin() * phi.sin() * omegay + theta.cos() * omegaz;
    omega2 = theta.cos() * phi.cos() * omegax +
             theta.cos() * phi.sin() * omegay - theta.sin() * omegaz;
    omega3 = -phi.sin() * omegax + phi.cos() * omegay;
  } else if (pcoord->options->type() == "gnomonic-equiangle") {
    int r = get_rank();
    auto layout = MeshBlockImpl::get_layout();
    auto [rx, ry, face_id] = layout->loc_of(r);
    auto face = CS_FACE_NAMES[face_id];

    auto alpha = mesh[1];
    auto beta = mesh[0];
    auto [lon, lat] = cs_ab_to_lonlat(face, alpha, beta);

    auto theta = M_PI / 2. - lat;
    auto phi = lon;

    omega1 = theta.sin() * phi.cos() * omegax +
             theta.sin() * phi.sin() * omegay + theta.cos() * omegaz;
    /*omega2 = theta.cos() * phi.cos() * omegax +
             theta.cos() * phi.sin() * omegay - theta.sin() * omegaz;
    omega3 = -phi.sin() * omegax + phi.cos() * omegay;*/

    // 2D coriolis only
    omega2 = torch::zeros_like(omega1);
    omega3 = torch::zeros_like(omega1);
  } else {
    throw std::runtime_error("CoriolisXYZ: unsupported coordinate system");
  }

  register_buffer("omega1", omega1);
  register_buffer("omega2", omega2);
  register_buffer("omega3", omega3);
}

torch::Tensor CoriolisXYZImpl::forward(torch::Tensor du, torch::Tensor w,
                                       torch::Tensor temp, double dt) {
  auto m1 = w[IDN] * w[IVX];
  auto m2 = w[IDN] * w[IVY];
  auto m3 = w[IDN] * w[IVZ];

  du[IVX] += 2. * dt * (omega3 * m2 - omega2 * m3);
  du[IVY] += 2. * dt * (omega1 * m3 - omega3 * m1);
  du[IVZ] += 2. * dt * (omega2 * m1 - omega1 * m2);

  return du;
}
}  // namespace snap
