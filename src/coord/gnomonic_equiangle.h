#pragma once

// snap
#include <snap/snap.h>

#include "../layout/cubed_sphere_constants.h"

namespace snap {

template <typename T>
inline DISPATCH_MACRO void gnomonic_ab_to_xyz(int face, T alpha, T beta, T* x,
                                              T* y, T* z) {
  T a = tan(alpha);
  T b = tan(beta);
  cs_face_xyz_from_tan(face, a, b, x, y, z);
  T r = sqrt((*x) * (*x) + (*y) * (*y) + (*z) * (*z));
  *x /= r;
  *y /= r;
  *z /= r;
}

template <typename T>
inline DISPATCH_MACRO void gnomonic_theta_phi(int face, T alpha, T beta,
                                              T* theta, T* phi) {
  T x, y, z;
  gnomonic_ab_to_xyz(face, alpha, beta, &x, &y, &z);
  z = max(T(-1), min(T(1), z));
  *theta = acos(z);
  *phi = atan2(y, x);
}

template <typename T>
inline DISPATCH_MACRO void gnomonic_sin_cos(int dim, T alpha, T beta,
                                            T* sin_theta, T* cos_theta) {
  (void)dim;
  T x = tan(alpha);
  T y = tan(beta);
  T C = sqrt(T(1) + x * x);
  T D = sqrt(T(1) + y * y);
  *cos_theta = -x * y / (C * D);
  *sin_theta = sqrt(T(1) + x * x + y * y) / (C * D);
}

template <typename T>
inline DISPATCH_MACRO void gnomonic_prim2local(T* w, int dim, T alpha, T beta) {
  T sin_theta, cos_theta;
  gnomonic_sin_cos(dim, alpha, beta, &sin_theta, &cos_theta);
  T u2 = w[IVY];
  T u3 = w[IVZ];
  if (dim == 2) {
    w[IVY] = sin_theta * u2;
    w[IVZ] = cos_theta * u2 + u3;
  } else {
    w[IVY] = u2 + cos_theta * u3;
    w[IVZ] = sin_theta * u3;
  }
}

template <typename T>
inline DISPATCH_MACRO void gnomonic_flux2global(T* flux, int dim, T alpha,
                                                T beta, int stride) {
  T sin_theta, cos_theta;
  gnomonic_sin_cos(dim, alpha, beta, &sin_theta, &cos_theta);
  T f2 = flux[IVY * stride];
  T f3 = flux[IVZ * stride];
  T ty, tz;
  if (dim == 2) {
    ty = f2 / sin_theta;
    tz = f3 - cos_theta * f2 / sin_theta;
  } else {
    ty = f2 - cos_theta * f3 / sin_theta;
    tz = f3 / sin_theta;
  }
  flux[IVY * stride] = ty + tz * cos_theta;
  flux[IVZ * stride] = tz + ty * cos_theta;
}

template <typename T>
inline DISPATCH_MACRO void gnomonic_contra_to_sph(T* w, int face, T alpha,
                                                  T beta) {
  T x = tan(alpha);
  T y = tan(beta);
  T delta = sqrt(x * x + y * y + 1);
  T C = sqrt(1 + x * x);
  T D = sqrt(1 + y * y);
  T vz = w[IVX], vx = w[IVY], vy = w[IVZ];
  T cart[3];
  auto l2c1 = cs_local_to_cart_vel(face, VEL1);
  auto l2c2 = cs_local_to_cart_vel(face, VEL2);
  auto l2c3 = cs_local_to_cart_vel(face, VEL3);
  cart[l2c1.idx] = l2c1.sgn * (vz - vx * x / D - vy * y / C) / delta;
  cart[l2c2.idx] = l2c2.sgn * (vz * x + vx * D - (vy * x * y) / C) / delta;
  cart[l2c3.idx] = l2c3.sgn * (vz * y + vy * C - (vx * x * y) / D) / delta;

  T theta, phi;
  gnomonic_theta_phi(face, alpha, beta, &theta, &phi);
  T st = sin(theta), ct = cos(theta), sp = sin(phi), cp = cos(phi);
  T cx = cart[0], cy = cart[1], cz = cart[2];
  w[IVX] = cx * st * cp + cy * st * sp + cz * ct;
  w[IVY] = cx * ct * cp + cy * ct * sp - cz * st;
  w[IVZ] = -cx * sp + cy * cp;
}

template <typename T>
inline DISPATCH_MACRO void gnomonic_sph_to_contra(T* w, int face, T alpha,
                                                  T beta) {
  T theta, phi;
  gnomonic_theta_phi(face, alpha, beta, &theta, &phi);
  T st = sin(theta), ct = cos(theta), sp = sin(phi), cp = cos(phi);
  T vr = w[IVX], vt = w[IVY], vp = w[IVZ];
  T cart[3];
  cart[0] = vr * st * cp + vt * ct * cp - vp * sp;
  cart[1] = vr * st * sp + vt * ct * sp + vp * cp;
  cart[2] = vr * ct - vt * st;

  T x = tan(alpha);
  T y = tan(beta);
  T delta = sqrt(x * x + y * y + 1);
  T C = sqrt(1 + x * x);
  T D = sqrt(1 + y * y);
  T local[3];
  auto c2l1 = cs_cart_to_local_vel(face, VEL1);
  auto c2l2 = cs_cart_to_local_vel(face, VEL2);
  auto c2l3 = cs_cart_to_local_vel(face, VEL3);
  local[c2l1.idx] = c2l1.sgn * cart[0];
  local[c2l2.idx] = c2l2.sgn * cart[1];
  local[c2l3.idx] = c2l3.sgn * cart[2];
  T vz = local[0], vx = local[1], vy = local[2];
  w[IVX] = (vz + x * vx + y * vy) / delta;
  w[IVY] = D / delta * (vx - x * vz);
  w[IVZ] = C / delta * (vy - y * vz);
}

}  // namespace snap
