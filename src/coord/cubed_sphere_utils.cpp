// C/C++
#include <cfloat>
#include <cmath>
#include <cstdint>
#include <cstring>

// snap
#include <snap/snap.h>

#include <snap/layout/cubed_sphere_layout.hpp>
#include <snap/mesh/meshblock.hpp>

#include "cubed_sphere_utils.hpp"
#include "spherical_utils.hpp"

namespace snap {
namespace {

int cs_face_id(char const *face) {
  for (int n = 0; n < 6; ++n) {
    if (strcmp(face, CS_FACE_NAMES[n]) == 0) return n;
  }
  throw std::runtime_error("invalid cubed-sphere face name");
}

}  // namespace

/*!
 * beta   ^ (T,3)
 *        |------
 *  (L,0) |  X  | (R,1)
 *        --------> alpha
 *            (B,2)
 * Convert angle on a face to a fractional center index (for 1-D interp).
 * beta: varies in Y (L/R)
 * alpha: varies in X (B/T)
 * Returns u in "cell-center units": 0.0 ~ center 0, 1.0 ~ center 1, ... (N-1)
 */
static inline double cs_angle_to_center_u(double angle, int N) {
  double d = M_PI / (2.0 * (double)N);
  // centers: angle = -pi/4 + (i+0.5)*d  => i = (angle + pi/4)/d - 0.5
  return (angle + M_PI / 4.0) / d - 0.5;
}

Vec3 cs_ab_to_xyz(char const *face, double alpha, double beta) {
  double a = tan(alpha), b = tan(beta);
  double x, y, z;
  cs_face_xyz_from_tan(cs_face_id(face), a, b, &x, &y, &z);
  return unit_vec3(x, y, z);
}

void cs_xyz_to_ab(char const *face, Vec3 v, double *alpha, double *beta) {
  if (strcmp(face, "+X") == 0) {
    *alpha = atan2(v.y, v.x);
    *beta = atan2(v.z, v.x);
  } else if (strcmp(face, "-X") == 0) {
    *alpha = atan2(-v.y, -v.x);
    *beta = atan2(v.z, -v.x);
  } else if (strcmp(face, "+Y") == 0) {
    *alpha = atan2(-v.x, v.y);
    *beta = atan2(v.z, v.y);
  } else if (strcmp(face, "-Y") == 0) {
    *alpha = atan2(v.x, -v.y);
    *beta = atan2(v.z, -v.y);
  } else if (strcmp(face, "+Z") == 0) {
    *alpha = atan2(v.y, v.z);
    *beta = atan2(-v.x, v.z);
  } else if (strcmp(face, "-Z") == 0) {
    *alpha = atan2(v.y, -v.z);
    *beta = atan2(v.x, -v.z);
  } else {
    throw std::runtime_error("cs_xyz_to_ab: invalid face name");
  }
}

void cs_build_ghost_usrc(double *usrc, int N, int nghost, int face_t,
                         int side_t) {
  const CSEdge emap = CS_FACE_EDGES[face_t][side_t];

  // which angle varies on target face
  const int src_ax = cs_side_axis(emap.nside);
  for (int depth = 1; depth <= nghost; ++depth) {
    for (int j = 0; j < N; ++j) {
      // 1) Target ghost center
      double alpha_t, beta_t;
      cs_equ_ghost_center(side_t, N, j, depth, &alpha_t, &beta_t);

      // 2) To cartesian
      Vec3 v = cs_ab_to_xyz(CS_FACE_NAMES[face_t], alpha_t, beta_t);

      // 3) Re-express on the source face from connectivity
      double alpha_s, beta_s;
      cs_xyz_to_ab(CS_FACE_NAMES[emap.nface], v, &alpha_s, &beta_s);

      // 4) Fractional abscissa along source edge’s interior line
      double u_src =
          cs_angle_to_center_u((src_ax == AX_Y) ? beta_s : alpha_s, N);

      // 5) Flip per connectivity flag
      if (emap.rev) u_src = (double)(N - 1) - u_src;

      // 6) Write out
      usrc[(depth - 1) * N + j] = u_src;
    }
  }
}

std::pair<torch::Tensor, torch::Tensor> cs_ab_to_lonlat(char const *face,
                                                        torch::Tensor alpha,
                                                        torch::Tensor beta) {
  auto x = alpha.tan();
  auto y = beta.tan();
  auto r = (x * x + y * y + 1).sqrt();

  torch::Tensor lon, lat;

  if (strcmp(face, "+X") == 0) {
    lon = alpha.clone();
    lat = (y / (1.0 + x * x).sqrt()).atan();
  } else if (strcmp(face, "+Y") == 0) {
    lon = alpha + 0.5 * M_PI;
    lat = (y / (1.0 + x * x).sqrt()).atan();
  } else if (strcmp(face, "-X") == 0) {
    lon = alpha + M_PI;
    lat = (y / (1.0 + x * x).sqrt()).atan();
  } else if (strcmp(face, "-Y") == 0) {
    lon = alpha + 1.5 * M_PI;
    lat = (y / (1.0 + x * x).sqrt()).atan();
  } else if (strcmp(face, "+Z") == 0) {
    lon = torch::atan2(x, -y);
    lat = torch::asin(1. / r);
  } else if (strcmp(face, "-Z") == 0) {
    lon = torch::atan2(x, y);
    lat = -torch::asin(1. / r);
  }

  // Map to the interval [0, 2 pi]
  lon += torch::where(lon < 0.0, 2.0 * M_PI, torch::zeros_like(lon));

  return {lon, lat};
}

static std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
cs_ab_to_xyz_tensors(char const *face, torch::Tensor const &alpha,
                     torch::Tensor const &beta) {
  auto a = alpha.tan();
  auto b = beta.tan();
  auto one = torch::ones_like(a);

  torch::Tensor x, y, z;
  if (strcmp(face, "+X") == 0) {
    x = one;
    y = a;
    z = b;
  } else if (strcmp(face, "+Y") == 0) {
    x = -a;
    y = one;
    z = b;
  } else if (strcmp(face, "-X") == 0) {
    x = -one;
    y = -a;
    z = b;
  } else if (strcmp(face, "+Z") == 0) {
    x = -b;
    y = a;
    z = one;
  } else if (strcmp(face, "-Y") == 0) {
    x = a;
    y = -one;
    z = b;
  } else if (strcmp(face, "-Z") == 0) {
    x = b;
    y = a;
    z = -one;
  } else {
    TORCH_CHECK(false, "invalid cubed-sphere face name: ", face);
  }

  auto r = (x * x + y * y + z * z).sqrt();
  return {x / r, y / r, z / r};
}

static std::pair<torch::Tensor, torch::Tensor> cs_ab_to_theta_phi(
    torch::Tensor const &alpha, torch::Tensor const &beta, int face_id) {
  auto [x, y, z] = cs_ab_to_xyz_tensors(CS_FACE_NAMES[face_id], alpha, beta);
  auto theta = z.clamp(-1.0, 1.0).acos();
  auto phi = torch::atan2(y, x);
  return {theta, phi};
}

//! Transform global cartesian velocity to local panel contravariant velocity
void cs_cart_to_contra_(torch::Tensor const &vel, torch::Tensor alpha,
                        torch::Tensor beta, int face_id) {
  auto x = alpha.tan();
  auto y = beta.tan();

  auto delta = sqrt(x * x + y * y + 1);
  auto C = sqrt(1 + x * x);
  auto D = sqrt(1 + y * y);

  std::array<torch::Tensor, 3> local_vel;

  auto const g2l = CS_CART_TO_LOCAL_VEL;
  auto f = face_id;

  local_vel[g2l[f][VEL1].idx] = g2l[f][VEL1].sgn * vel[VEL1];
  local_vel[g2l[f][VEL2].idx] = g2l[f][VEL2].sgn * vel[VEL2];
  local_vel[g2l[f][VEL3].idx] = g2l[f][VEL3].sgn * vel[VEL3];

  auto vz = local_vel[VEL1];
  auto vx = local_vel[VEL2];
  auto vy = local_vel[VEL3];

  vel[VEL1] = (vz + x * vx + y * vy) / delta;
  vel[VEL2] = D / delta * (vx - x * vz);
  vel[VEL3] = C / delta * (vy - y * vz);
}

void cs_contra_to_cart_(torch::Tensor const &vel, torch::Tensor alpha,
                        torch::Tensor beta, int face_id) {
  auto x = alpha.tan();
  auto y = beta.tan();

  auto delta = sqrt(x * x + y * y + 1);
  auto C = sqrt(1 + x * x);
  auto D = sqrt(1 + y * y);

  auto const l2g = CS_LOCAL_TO_CART_VEL;
  auto f = face_id;

  auto vz = vel[VEL1].clone();
  auto vx = vel[VEL2].clone();
  auto vy = vel[VEL3].clone();

  vel[l2g[f][VEL1].idx] =
      l2g[f][VEL1].sgn * (vz - vx * x / D - vy * y / C) / delta;
  vel[l2g[f][VEL2].idx] =
      l2g[f][VEL2].sgn * (vz * x + vx * D - (vy * x * y) / C) / delta;
  vel[l2g[f][VEL3].idx] =
      l2g[f][VEL3].sgn * (vz * y + vy * C - (vx * x * y) / D) / delta;
}

void cs_contra_to_sph_(torch::Tensor const &vel, torch::Tensor alpha,
                       torch::Tensor beta, int face_id) {
  auto [theta, phi] = cs_ab_to_theta_phi(alpha, beta, face_id);
  cs_contra_to_cart_(vel, alpha, beta, face_id);
  sph_cart_to_contra_(vel, theta, phi);
}

void cs_sph_to_contra_(torch::Tensor const &vel, torch::Tensor alpha,
                       torch::Tensor beta, int face_id) {
  auto [theta, phi] = cs_ab_to_theta_phi(alpha, beta, face_id);
  sph_contra_to_cart_(vel, theta, phi);
  cs_cart_to_contra_(vel, alpha, beta, face_id);
}

}  // namespace snap
