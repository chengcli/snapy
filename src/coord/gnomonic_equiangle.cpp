// snap
#include <snap/snap.h>

#include "coordinate.hpp"

namespace snap {

void GnomonicEquiangleImpl::reset() {
  auto const &op = options;

  // dimension 1
  auto dx = (op.x1max() - op.x1min()) / op.nx1();
  dx1f = register_buffer("dx1f", torch::ones(op.nc1()) * dx);
  x1v = register_buffer(
      "x1v", 0.5 * (x1f.slice(0, 0, op.nc1()) + x1f.slice(0, 1, op.nc1() + 1)));

  // dimension 2
  dx = (op.x2max() - op.x2min()) / op.nx2();
  dx2f = register_buffer("dx2f", torch::ones(op.nc2()) * dx);
  x2v = register_buffer(
      "x2v", 0.5 * (x2f.slice(0, 0, op.nc2()) + x2f.slice(0, 1, op.nc2() + 1)));

  // dimension 3
  dx = (op.x3max() - op.x3min()) / op.nx3();
  dx3f = register_buffer("dx3f", torch::ones(op.nc3()) * dx);
  x3v = register_buffer(
      "x3v", 0.5 * (x3f.slice(0, 0, op.nc3()) + x3f.slice(0, 1, op.nc3() + 1)));

  // register buffers defined in the base class
  register_buffer("x1f", x1f);
  register_buffer("x2f", x2f);
  register_buffer("x3f", x3f);

  // populate and register geometry data
  auto x = x2v.tan().unsqueeze(0);
  auto xf = x2f.tan().unsqueeze(0);
  auto y = x3v.tan().unsqueeze(-1);
  auto yf = x3f.tan().unsqueeze(-1);

  auto C = (1.0 + x * x).sqrt();
  auto Cf = (1.0 + xf * xf).sqrt();
  auto D = (1.0 + y * y).sqrt();
  auto Df = (1.0 + yf * yf).sqrt();

  cosine_cell_kj = -x * y / (C * D);
  sine_cell_kj = (1.0 + x * x + y * y).sqrt() / (C * D);

  cosine_face2_kj = -xf * y / (Cf * D);
  sine_face2_kj = (1.0 + xf * xf + y * y).sqrt() / (Cf * D);

  cosine_face3_kj = -x * yf / (C * Df);
  sine_face3_kj = (1.0 + x * x + yf * yf).sqrt() / (C * Df);

  register_buffer("cosine_cell_kj", cosine_cell_kj);
  register_buffer("sine_cell_kj", sine_cell_kj);
  register_buffer("cosine_face2_kj", cosine_face2_kj);
  register_buffer("sine_face2_kj", sine_face2_kj);
  register_buffer("cosine_face3_kj", cosine_face3_kj);
  register_buffer("sine_face3_kj", sine_face3_kj);

  auto x1 = x2f.tan().unsqueeze(0);
  auto x2 = x2f.tan().roll(-1).unsqueeze(0);
  auto delta1 = (1. + x1 * x1 + y * y).sqrt();
  auto delta2 = (1. + x2 * x2 + y * y).sqrt();
  auto delta1f = (1. + x1 * x1 + yf * yf).sqrt();
  auto delta2f = (1. + x2 * x2 + yf * yf).sqrt();

  dx2f_ang_kj = ((1. + x1 * x2 + y * y) / (delta1 * delta2)).acos();
  dx2f_ang_face3_kj = ((1. + x1 * x2 + yf * yf) / (delta1f * delta2)).acos();

  register_buffer("dx2f_ang_kj", dx2f_ang_kj);
  register_buffer("dx2f_ang_face3_kj", dx2f_ang_face3_kj);

  auto y1 = x3f.tan().unsqueeze(-1);
  auto y2 = x3f.tan().roll(-1).unsqueeze(-1);
  delta1 = (1. + x * x + y1 * y1).sqrt();
  delta2 = (1. + x * x + y2 * y2).sqrt();
  delta1f = (1. + xf * xf + y1 * y1).sqrt();
  delta2f = (1. + xf * xf + y2 * y2).sqrt();

  dx3f_ang_kj = ((1. + x * x + y1 * y2) / (delta1 * delta2)).acos();
  dx3f_ang_face2_kj = ((1. + xf * xf + y1 * y2) / (delta1f * delta2)).acos();

  register_buffer("dx3f_ang_kj", dx3f_ang_kj);
  register_buffer("dx3f_ang_face2_kj", dx3f_ang_face2_kj);

  auto fx1 = face_area2() * sine_face2_kj.unsqueeze(-1);
  auto fx2 = fx1.roll(-1, /*dim=*/1);
  auto fy1 = face_area3() * sine_face3_kj.unsqueeze(-1);
  auto fy2 = fy1.roll(-1, /*dim=*/0);

  x_ov_rD_kji = (fx1 - fx2) / cell_volume();
  y_ov_rC_kji = (fy1 - fy2) / cell_volume();

  register_buffer("x_ov_rD_kji", x_ov_rD_kji);
  register_buffer("y_ov_rC_kji", y_ov_rC_kji);

  // register metric data
  auto vol = cell_volume();
  g11 = register_buffer("g11", torch::ones_like(vol));
  g22 = register_buffer("g22", torch::ones_like(vol));
  g33 = register_buffer("g33", torch::ones_like(vol));
  gi11 = register_buffer("gi11", torch::ones_like(vol));
  gi22 = register_buffer("gi22", torch::ones_like(vol));
  gi33 = register_buffer("gi33", torch::ones_like(vol));
  g12 = register_buffer("g12", torch::zeros_like(vol));
  g13 = register_buffer("g13", torch::zeros_like(vol));
  g23 = register_buffer("g23", torch::zeros_like(vol));
}

torch::Tensor GnomonicEquiangleImpl::face_area1() const {
  return (x1f * x1f).unsqueeze(0).unsqueeze(1) *
         (dx2f_ang_kj * dx3f_ang_kj * sine_cell_kj).unsqueeze(-1);
}

torch::Tensor GnomonicEquiangleImpl::face_area2() const {
  return (x1v * dx1f).unsqueeze(0).unsqueeze(1) *
         dx3f_ang_face2_kj.unsqueeze(-1);
}

torch::Tensor GnomonicEquiangleImpl::face_area3() const {
  return (x1v * dx1f).unsqueeze(0).unsqueeze(1) *
         dx2f_ang_face3_kj.unsqueeze(-1);
}

torch::Tensor GnomonicEquiangleImpl::cell_volume() const {
  return (x1v * x1v * dx1f).unsqueeze(0).unsqueeze(1) *
         (dx2f_ang_kj * dx3f_ang_kj * sine_cell_kj).unsqueeze(-1);
}

void GnomonicEquiangleImpl::vec_lower_(torch::Tensor &vel) const {
  auto v = vel[1].clone();
  auto w = vel[2].clone();
  vel[1] = v + w * cosine_cell_kj.unsqueeze(-1);
  vel[2] = w + v * cosine_cell_kj.unsqueeze(-1);
}

void GnomonicEquiangleImpl::vec_raise_(torch::Tensor &vel) const {
  auto v = vel[1].clone();
  auto w = vel[2].clone();
  auto cth = cosine_cell_kj.unsqueeze(-1);
  auto sth2 = 1. - cth * cth;

  vel[1] = v / sth2 - w * cth / sth2;
  vel[2] = -v * cth / sth2 + w / sth2;
}

// TODO(cli):: CHECK
void GnomonicEquiangleImpl::_set_face2_metric() const {
  auto cos_theta = cosine_face2_kj.unsqueeze(-1);
  auto sin_theta = sine_face2_kj.unsqueeze(-1);

  g11.set_(torch::ones_like(cos_theta));
  g22.set_(torch::ones_like(cos_theta));
  g23.set_(cos_theta);
  g33.set_(torch::ones_like(cos_theta));

  gi11.set_(torch::ones_like(cos_theta));
  gi22.set_(1. / (sin_theta * sin_theta));
  gi33.set_(1. / (sin_theta * sin_theta));
}

// TODO(cli):: CHECK
void GnomonicEquiangleImpl::_set_face3_metric() const {
  auto cos_theta = cosine_face3_kj.unsqueeze(-1);
  auto sin_theta = sine_face3_kj.unsqueeze(-1);

  g11.set_(torch::ones_like(cos_theta));
  g22.set_(torch::ones_like(cos_theta));
  g23.set_(cos_theta);
  g33.set_(torch::ones_like(cos_theta));

  gi11.set_(torch::ones_like(cos_theta));
  gi22.set_(1. / (sin_theta * sin_theta));
  gi33.set_(1. / (sin_theta * sin_theta));
}

void GnomonicEquiangleImpl::prim2local2_(torch::Tensor &w) const {
  _set_face2_metric();

  // Extract global projected 4-velocities
  auto uu1 = w[IVX];
  auto uu2 = w[IVY];
  auto uu3 = w[IVZ];

  // Calculate transformation matrix
  auto T11 = torch::ones_like(g11);
  auto T22 = 1.0 / gi22.sqrt();
  auto T32 = g23 / g33.sqrt();
  auto T33 = g33.sqrt();

  // Transform projected velocities
  auto ux = T11 * uu1;
  auto uy = T22 * uu2;
  auto uz = T32 * uu2 + T33 * uu3;

  // Set local projected 4-velocities
  w[IVX] = ux;
  w[IVY] = uy;
  w[IVZ] = uz;
}

void GnomonicEquiangleImpl::prim2local3_(torch::Tensor &w) const {
  _set_face3_metric();

  // Extract global projected 4-velocities
  auto uu1 = w[IVX];
  auto uu2 = w[IVY];
  auto uu3 = w[IVZ];

  // Calculate transformation matrix
  auto T11 = torch::ones_like(g11);
  auto T22 = g22.sqrt();
  auto T23 = g23 / g22.sqrt();
  auto T33 = 1.0 / gi33.sqrt();

  // Transform projected velocities
  auto ux = T11 * uu1;
  auto uy = T22 * uu2 + T23 * uu3;
  auto uz = T33 * uu3;

  // Set local projected 4-velocities
  w[IVX] = ux;
  w[IVY] = uy;
  w[IVZ] = uz;
}

// does not de-orthonormal, but only transforms to covariant form
void GnomonicEquiangleImpl::flux2global1_(torch::Tensor &flux) const {
  auto cos_theta = cosine_cell_kj.unsqueeze(-1);

  // Extract contravariant fluxes
  auto ty = flux[IVY].clone();
  auto tz = flux[IVZ].clone();

  // Transform to covariant fluxes
  flux[IVY] = ty + tz * cos_theta;
  flux[IVZ] = tz + ty * cos_theta;
}

// de-orthonormal and transforms to covariant form
void GnomonicEquiangleImpl::flux2global2_(torch::Tensor &flux) const {
  _set_face2_metric();

  // Extract local conserved quantities and fluxes
  auto txx = flux[IVX];
  auto txy = flux[IVY];
  auto txz = flux[IVZ];

  // Calculate transformation matrix
  auto T11 = 1.0;
  auto T22 = gi22.sqrt();
  auto T32 = -gi22.sqrt() * g23 / g33;
  auto T33 = 1.0 / g33.sqrt();

  // Set fluxes
  flux[IVX] = T11 * txx;
  flux[IVY] = T22 * txy;
  flux[IVZ] = T32 * txy + T33 * txz;

  auto cos_theta = cosine_face2_kj.unsqueeze(-1);

  // Extract contravariant fluxes
  auto ty = flux[IVY].clone();
  auto tz = flux[IVZ].clone();

  // Transform to covariant fluxes
  flux[IVY] = ty + tz * cos_theta;
  flux[IVZ] = tz + ty * cos_theta;
}

// de-orthonormal and transforms to covariant form
void GnomonicEquiangleImpl::flux2global3_(torch::Tensor &flux) const {
  _set_face3_metric();

  // Extract local conserved quantities and fluxes
  auto txx = flux[IVX];
  auto txy = flux[IVY];
  auto txz = flux[IVZ];

  // Calculate transformation matrix
  auto T11 = 1.0;
  auto T22 = 1.0 / g22.sqrt();
  auto T23 = -g23 / g22 * gi33.sqrt();
  auto T33 = gi33.sqrt();

  // Set fluxes
  flux[IVX] = T11 * txx;
  flux[IVY] = T22 * txy + T23 * txz;
  flux[IVZ] = T33 * txz;

  auto cos_theta = cosine_face3_kj.unsqueeze(-1);

  // Extract contravariant fluxes
  auto ty = flux[IVY].clone();
  auto tz = flux[IVZ].clone();

  // Transform to covariant fluxes
  flux[IVY] = ty + tz * cos_theta;
  flux[IVZ] = tz + ty * cos_theta;
}

torch::Tensor GnomonicEquiangleImpl::forward(torch::Tensor prim,
                                             torch::Tensor flux1,
                                             torch::Tensor flux2,
                                             torch::Tensor flux3) {
  auto div = CoordinateImpl::forward(prim, flux1, flux2, flux3);

  auto cosine = cosine_cell_kj.unsqueeze(-1);
  auto sine2 = sine_cell_kj.square().unsqueeze(-1);

  // General variables
  auto v1 = prim[IVX];
  auto v2 = prim[IVY];
  auto v3 = prim[IVZ];
  auto radius = x1v.unsqueeze(0).unsqueeze(1);

  torch::Tensor pr, rho;

  auto v_2 = v2 + v3 * cosine;
  auto v_3 = v3 + v2 * cosine;

  if (options.eos_type() == "shallow-water") {
    pr = 0.5 * prim[IDN].square();
    rho = prim[IDN];
  } else {
    pr = prim[IPR];
    rho = prim[IDN];
    // Update flux 1 (excluded from shallow water case)
    auto src1 = (2.0 * pr + rho * (v2 * v_2 + v3 * v_3)) / radius;
    div[IVX] -= src1;
  }

  // Update flux 2
  auto src2 =
      -x_ov_rD_kji * (pr + rho * v3 * v3 * sine2) - rho * v1 * v_2 / radius;
  div[IVY] -= src2;

  // Update flux 3
  auto src3 =
      -y_ov_rC_kji * (pr + rho * v2 * v2 * sine2) - rho * v1 * v_3 / radius;
  div[IVZ] -= src3;

  return div;
}

}  // namespace snap
