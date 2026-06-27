// torch
#include <ATen/Dispatch.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/csrc/distributed/c10d/symm_mem/CUDASymmetricMemory-inl.h>

// snap
#include <snap/snap.h>

#include "../recon/interp_impl.cuh"
#include "../riemann/hllc_impl.h"
#include "../riemann/lmars_impl.h"
#include "fused_recon_riemann_dispatch.hpp"

namespace snap {
namespace {

template <typename T>
__device__ T sqr(T x) {
  return x * x;
}

template <typename T>
__device__ T interp(T const* line, int v, int start, int axis_size,
                    FusedReconScheme scheme, bool right) {
  if (scheme == FusedReconScheme::CP3) {
    return interp_shared_poly3_impl(line, v, start, axis_size, right);
  }
  if (scheme == FusedReconScheme::CP5) {
    return interp_shared_poly5_impl(line, v, start, axis_size, right);
  }
  if (scheme == FusedReconScheme::WENO3) {
    return interp_shared_weno3_impl(line, v, start, axis_size, right);
  }
  return interp_shared_weno5_impl(line, v, start, axis_size, right);
}

template <typename T>
__device__ T ideal_moist_feps(T const* y, int ny, T const* inv_mu_ratio_m1) {
  T out = 1.;
  for (int n = 0; n < ny; ++n) out += y[n] * inv_mu_ratio_m1[n];
  return out;
}

template <typename T>
__device__ T ideal_moist_fsig(T const* y, int ny, T const* cv_ratio_m1) {
  T out = 1.;
  for (int n = 0; n < ny; ++n) out += y[n] * cv_ratio_m1[n];
  return out;
}

template <typename T>
__device__ void eos_side_quantities(T const* wl, T const* wr, int ny,
                                    FusedEos eos, T gammad,
                                    T const* inv_mu_ratio_m1,
                                    T const* cv_ratio_m1, T const* u0, T* el,
                                    T* er, T* gl, T* gr, T* cl, T* cr) {
  if (eos == FusedEos::IdealGas) {
    *el = wl[IPR] / (gammad - 1.);
    *er = wr[IPR] / (gammad - 1.);
    *gl = gammad;
    *gr = gammad;
  } else {
    T yl[32], yr[32];
    T suml = 0., sumr = 0.;
    for (int n = 0; n < ny; ++n) {
      yl[n] = wl[ICY + n];
      yr[n] = wr[ICY + n];
      suml += yl[n];
      sumr += yr[n];
    }
    T fepsl = ideal_moist_feps(yl, ny, inv_mu_ratio_m1);
    T fepsr = ideal_moist_feps(yr, ny, inv_mu_ratio_m1);
    T fsigl = ideal_moist_fsig(yl, ny, cv_ratio_m1);
    T fsigr = ideal_moist_fsig(yr, ny, cv_ratio_m1);
    *el = wl[IPR] * fsigl / fepsl / (gammad - 1.);
    *er = wr[IPR] * fsigr / fepsr / (gammad - 1.);
    *el += wl[IDN] * (1. - suml) * u0[0];
    *er += wr[IDN] * (1. - sumr) * u0[0];
    for (int n = 0; n < ny; ++n) {
      *el += wl[IDN] * yl[n] * u0[1 + n];
      *er += wr[IDN] * yr[n] * u0[1 + n];
    }
    *gl = 1. + (gammad - 1.) * fepsl / fsigl;
    *gr = 1. + (gammad - 1.) * fepsr / fsigr;
  }
  *cl = sqrt((*gl) * wl[IPR] / wl[IDN]);
  *cr = sqrt((*gr) * wr[IPR] / wr[IDN]);
}

template <typename T>
__global__ void fused_kernel(T const* w, T* flux, int nvar, int nc3, int nc2,
                             int nc1, int dim, FusedReconScheme recon_prim,
                             FusedReconScheme recon_vel,
                             FusedRiemannSolver solver, FusedEos eos,
                             T gammad, T density_floor, T pressure_floor,
                             bool eos_limiter, T const* inv_mu_ratio_m1,
                             T const* cv_ratio_m1, T const* u0) {
  int ncells = nc1 * nc2 * nc3;
  int axis_size = dim == 3 ? nc1 : (dim == 2 ? nc2 : nc3);
  int stride_dim = dim == 3 ? 1 : (dim == 2 ? nc1 : nc1 * nc2);
  int stride_var = ncells;
  int pos = threadIdx.x;
  int line = blockIdx.x;
  int base = 0;

  if (dim == 3) {
    int j = line % nc2;
    int k = line / nc2;
    base = k * nc2 * nc1 + j * nc1;
  } else if (dim == 2) {
    int i = line % nc1;
    int k = line / nc1;
    base = k * nc2 * nc1 + i;
  } else {
    int i = line % nc1;
    int j = line / nc1;
    base = j * nc1 + i;
  }

  extern __shared__ unsigned char memory[];
  T* smem = reinterpret_cast<T*>(memory);
  if (pos < axis_size) {
    int flat = base + pos * stride_dim;
    for (int v = 0; v < nvar; ++v) {
      smem[v * axis_size + pos] = w[v * stride_var + flat];
    }
  }
  __syncthreads();

  if (pos >= axis_size) return;
  int flat = base + pos * stride_dim;
  int nghost = (recon_prim == FusedReconScheme::CP3 ||
                recon_prim == FusedReconScheme::WENO3)
                   ? 2
                   : 3;
  int il = nghost;
  int iu = axis_size - nghost;

  if (pos < il || pos > iu + 1) {
    for (int v = 0; v < nvar; ++v) flux[v * stride_var + flat] = 0.;
    return;
  }

  int wl_start = pos - il;
  int wr_start = pos - (il - 1);
  T wl_local[64], wr_local[64];
  for (int v = 0; v < nvar; ++v) {
    auto scheme = (v == IDN || v >= ICY) ? recon_prim : recon_vel;
    wl_local[v] =
        interp(smem, v, wl_start, axis_size, scheme, /*right=*/true);
    wr_local[v] =
        interp(smem, v, wr_start, axis_size, scheme, /*right=*/false);
  }

  if (eos_limiter) {
    wl_local[IDN] = max(wl_local[IDN], density_floor);
    wr_local[IDN] = max(wr_local[IDN], density_floor);
    wl_local[IPR] = max(wl_local[IPR], pressure_floor);
    wr_local[IPR] = max(wr_local[IPR], pressure_floor);
    for (int v = ICY; v < nvar; ++v) {
      wl_local[v] = max(wl_local[v], T(0.));
      wr_local[v] = max(wr_local[v], T(0.));
    }
  }

  T el = 0., er = 0., gl = 0., gr = 0., cl = 0., cr = 0.;
  int ny = nvar - ICY;
  eos_side_quantities(wl_local, wr_local, ny, eos, gammad, inv_mu_ratio_m1,
                      cv_ratio_m1, u0, &el, &er, &gl, &gr, &cl, &cr);

  if (solver == FusedRiemannSolver::LMARS) {
    lmars_impl(flux + flat, wl_local, wr_local, el / wl_local[IDN],
               er / wr_local[IDN], gl, gr, dim, ny, stride_var);
  } else {
    hllc_impl(flux + flat, wl_local, wr_local, el, er, gl, gr, cl, cr, dim, ny,
              stride_var);
  }
}

enum {
  CS_SIDE_L = 0,
  CS_SIDE_R = 1,
  CS_SIDE_B = 2,
  CS_SIDE_T = 3,
  CS_META_ENABLED = 0,
  CS_META_PEER_RANK = 1,
  CS_META_PEER_SIDE = 2,
  CS_META_REV = 3,
  CS_META_STRIDE = 4,
};

__device__ __constant__ int kCsLocalToCartIdx[6][3] = {
    {1, 2, 0}, {2, 1, 0}, {1, 2, 0},
    {0, 2, 1}, {2, 1, 0}, {0, 2, 1}};
__device__ __constant__ int kCsLocalToCartSgn[6][3] = {
    {+1, +1, +1}, {+1, -1, +1}, {-1, -1, +1},
    {+1, +1, -1}, {-1, +1, +1}, {-1, +1, +1}};
__device__ __constant__ int kCsCartToLocalIdx[6][3] = {
    {2, 0, 1}, {2, 1, 0}, {2, 0, 1},
    {0, 2, 1}, {2, 1, 0}, {0, 2, 1}};
__device__ __constant__ int kCsCartToLocalSgn[6][3] = {
    {+1, +1, +1}, {+1, -1, +1}, {-1, -1, +1},
    {+1, -1, +1}, {-1, +1, +1}, {-1, +1, +1}};

template <typename T>
__device__ void cs_ab_to_xyz(int face, T alpha, T beta, T* x, T* y, T* z) {
  T a = tan(alpha);
  T b = tan(beta);
  if (face == 0) {
    *x = 1;
    *y = a;
    *z = b;
  } else if (face == 1) {
    *x = -a;
    *y = 1;
    *z = b;
  } else if (face == 2) {
    *x = -1;
    *y = -a;
    *z = b;
  } else if (face == 3) {
    *x = -b;
    *y = a;
    *z = 1;
  } else if (face == 4) {
    *x = a;
    *y = -1;
    *z = b;
  } else {
    *x = b;
    *y = a;
    *z = -1;
  }
  T r = sqrt((*x) * (*x) + (*y) * (*y) + (*z) * (*z));
  *x /= r;
  *y /= r;
  *z /= r;
}

template <typename T>
__device__ void cs_theta_phi(int face, T alpha, T beta, T* theta, T* phi) {
  T x, y, z;
  cs_ab_to_xyz(face, alpha, beta, &x, &y, &z);
  z = max(T(-1), min(T(1), z));
  *theta = acos(z);
  *phi = atan2(y, x);
}

template <typename T>
__device__ void cs_contra_to_sph(T* w, int face, T alpha, T beta) {
  T x = tan(alpha);
  T y = tan(beta);
  T delta = sqrt(x * x + y * y + 1);
  T C = sqrt(1 + x * x);
  T D = sqrt(1 + y * y);
  T vz = w[IVX], vx = w[IVY], vy = w[IVZ];
  T cart[3];
  cart[kCsLocalToCartIdx[face][VEL1]] =
      kCsLocalToCartSgn[face][VEL1] * (vz - vx * x / D - vy * y / C) / delta;
  cart[kCsLocalToCartIdx[face][VEL2]] =
      kCsLocalToCartSgn[face][VEL2] * (vz * x + vx * D - (vy * x * y) / C) /
      delta;
  cart[kCsLocalToCartIdx[face][VEL3]] =
      kCsLocalToCartSgn[face][VEL3] * (vz * y + vy * C - (vx * x * y) / D) /
      delta;

  T theta, phi;
  cs_theta_phi(face, alpha, beta, &theta, &phi);
  T st = sin(theta), ct = cos(theta), sp = sin(phi), cp = cos(phi);
  T cx = cart[0], cy = cart[1], cz = cart[2];
  w[IVX] = cx * st * cp + cy * st * sp + cz * ct;
  w[IVY] = cx * ct * cp + cy * ct * sp - cz * st;
  w[IVZ] = -cx * sp + cy * cp;
}

template <typename T>
__device__ void cs_sph_to_contra(T* w, int face, T alpha, T beta) {
  T theta, phi;
  cs_theta_phi(face, alpha, beta, &theta, &phi);
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
  local[kCsCartToLocalIdx[face][VEL1]] =
      kCsCartToLocalSgn[face][VEL1] * cart[0];
  local[kCsCartToLocalIdx[face][VEL2]] =
      kCsCartToLocalSgn[face][VEL2] * cart[1];
  local[kCsCartToLocalIdx[face][VEL3]] =
      kCsCartToLocalSgn[face][VEL3] * cart[2];
  T vz = local[0], vx = local[1], vy = local[2];
  w[IVX] = (vz + x * vx + y * vy) / delta;
  w[IVY] = D / delta * (vx - x * vz);
  w[IVZ] = C / delta * (vy - y * vz);
}

template <typename T>
__device__ void cs_coords(int side, int edge_pos, int i, int nc3, int nc2,
                          int nc1, int nghost, T const* x2v, T const* x2f,
                          T const* x3v, T const* x3f, T* alpha, T* beta,
                          int* k, int* j, int* face_pos) {
  (void)i;
  if (side == CS_SIDE_L || side == CS_SIDE_R) {
    *k = edge_pos;
    *j = side == CS_SIDE_L ? nghost : nc2 - nghost - 1;
    *face_pos = side == CS_SIDE_L ? nghost : nc2 - nghost + 1;
    *alpha = x2f[side == CS_SIDE_L ? nghost : nc2 - nghost];
    *beta = x3v[edge_pos];
  } else {
    *k = side == CS_SIDE_B ? nghost : nc3 - nghost - 1;
    *j = edge_pos;
    *face_pos = side == CS_SIDE_B ? nghost : nc3 - nghost + 1;
    *alpha = x2v[edge_pos];
    *beta = x3f[side == CS_SIDE_B ? nghost : nc3 - nghost];
  }
}

template <typename T>
__global__ void cs_pack_kernel(T const* w, T* buf, int const* side_meta,
                               int nvar, int nc3, int nc2, int nc1, int face,
                               FusedReconScheme recon_prim,
                               FusedReconScheme recon_vel, T density_floor,
                               T pressure_floor, bool eos_limiter,
                               T const* x2v, T const* x2f, T const* x3v,
                               T const* x3f) {
  int edge_len = max(nc2, nc3);
  int line = blockIdx.x;
  int i = line % nc1;
  int edge_pos = (line / nc1) % edge_len;
  int side = line / (nc1 * edge_len);
  if (!side_meta[side * CS_META_STRIDE + CS_META_ENABLED]) return;
  if ((side <= CS_SIDE_R && edge_pos >= nc3) ||
      (side >= CS_SIDE_B && edge_pos >= nc2)) {
    return;
  }

  int dim = side <= CS_SIDE_R ? 2 : 1;
  int axis_size = dim == 2 ? nc2 : nc3;
  int stride_dim = dim == 2 ? nc1 : nc1 * nc2;
  int stride_var = nc1 * nc2 * nc3;
  int pos = threadIdx.x;
  int base = 0;
  if (side <= CS_SIDE_R) {
    base = edge_pos * nc2 * nc1 + i;
  } else {
    base = edge_pos * nc1 + i;
  }

  extern __shared__ unsigned char memory[];
  T* smem = reinterpret_cast<T*>(memory);
  if (pos < axis_size) {
    int in = base + pos * stride_dim;
    for (int v = 0; v < nvar; ++v) {
      smem[v * axis_size + pos] = w[v * stride_var + in];
    }
  }
  __syncthreads();

  int nghost = (recon_prim == FusedReconScheme::CP3 ||
                recon_prim == FusedReconScheme::WENO3)
                   ? 2
                   : 3;
  int k, j, face_pos;
  T alpha, beta;
  cs_coords(side, edge_pos, i, nc3, nc2, nc1, nghost, x2v, x2f, x3v, x3f,
            &alpha, &beta, &k, &j, &face_pos);
  if (pos != face_pos) return;
  int il = nghost;
  bool send_right_state = (side == CS_SIDE_L || side == CS_SIDE_B);
  int start = send_right_state ? face_pos - (il - 1) : face_pos - il;
  bool right_interp = send_right_state ? false : true;

  T local[64];
  for (int v = 0; v < nvar; ++v) {
    auto scheme = (v == IDN || v >= ICY) ? recon_prim : recon_vel;
    local[v] = interp(smem, v, start, axis_size, scheme, right_interp);
  }
  if (eos_limiter) {
    local[IDN] = max(local[IDN], density_floor);
    local[IPR] = max(local[IPR], pressure_floor);
    for (int v = ICY; v < nvar; ++v) local[v] = max(local[v], T(0));
  }
  cs_contra_to_sph(local, face, alpha, beta);

  int buf_stride_var = edge_len * nc1;
  int out = ((side * nvar) * edge_len + edge_pos) * nc1 + i;
  for (int v = 0; v < nvar; ++v) buf[out + v * buf_stride_var] = local[v];
}

__global__ void cs_sync_kernel(uint32_t** signal_pads, int rank,
                               int world_size) {
  c10d::symmetric_memory::sync_remote_blocks<true, true>(
      signal_pads, rank, world_size);
}

template <typename T>
__global__ void cs_flux_kernel(T const* w, T* flux2, T* flux3, void** buf_ptrs,
                               int const* side_meta, int nvar, int nc3,
                               int nc2, int nc1, int face,
                               FusedReconScheme recon_prim,
                               FusedReconScheme recon_vel,
                               FusedRiemannSolver solver, FusedEos eos,
                               T gammad, T density_floor, T pressure_floor,
                               bool eos_limiter, T const* inv_mu_ratio_m1,
                               T const* cv_ratio_m1, T const* u0,
                               T const* x2v, T const* x2f, T const* x3v,
                               T const* x3f) {
  int edge_len = max(nc2, nc3);
  int line = blockIdx.x;
  int i = line % nc1;
  int edge_pos = (line / nc1) % edge_len;
  int side = line / (nc1 * edge_len);
  if (!side_meta[side * CS_META_STRIDE + CS_META_ENABLED]) return;
  if ((side <= CS_SIDE_R && edge_pos >= nc3) ||
      (side >= CS_SIDE_B && edge_pos >= nc2)) {
    return;
  }

  int peer_rank = side_meta[side * CS_META_STRIDE + CS_META_PEER_RANK];
  int peer_side = side_meta[side * CS_META_STRIDE + CS_META_PEER_SIDE];
  int rev = side_meta[side * CS_META_STRIDE + CS_META_REV];
  int peer_edge = rev ? ((side <= CS_SIDE_R ? nc3 : nc2) - 1 - edge_pos)
                      : edge_pos;
  auto peer_buf = static_cast<T const*>(buf_ptrs[peer_rank]);
  int buf_stride_var = edge_len * nc1;
  int remote_off = ((peer_side * nvar) * edge_len + peer_edge) * nc1 + i;

  int dim = side <= CS_SIDE_R ? 2 : 1;
  int axis_size = dim == 2 ? nc2 : nc3;
  int stride_dim = dim == 2 ? nc1 : nc1 * nc2;
  int stride_var = nc1 * nc2 * nc3;
  int pos = threadIdx.x;
  int base = 0;
  if (side <= CS_SIDE_R) {
    base = edge_pos * nc2 * nc1 + i;
  } else {
    base = edge_pos * nc1 + i;
  }

  extern __shared__ unsigned char memory[];
  T* smem = reinterpret_cast<T*>(memory);
  if (pos < axis_size) {
    int in = base + pos * stride_dim;
    for (int v = 0; v < nvar; ++v) {
      smem[v * axis_size + pos] = w[v * stride_var + in];
    }
  }
  __syncthreads();

  int nghost = (recon_prim == FusedReconScheme::CP3 ||
                recon_prim == FusedReconScheme::WENO3)
                   ? 2
                   : 3;
  int k, j, face_pos;
  T alpha, beta;
  cs_coords(side, edge_pos, i, nc3, nc2, nc1, nghost, x2v, x2f, x3v, x3f,
            &alpha, &beta, &k, &j, &face_pos);
  if (pos != face_pos) return;
  int il = nghost;
  bool own_is_right_state = (side == CS_SIDE_L || side == CS_SIDE_B);
  int start = own_is_right_state ? face_pos - (il - 1) : face_pos - il;
  bool right_interp = own_is_right_state ? false : true;

  T own[64], remote[64], wl[64], wr[64];
  for (int v = 0; v < nvar; ++v) {
    auto scheme = (v == IDN || v >= ICY) ? recon_prim : recon_vel;
    own[v] = interp(smem, v, start, axis_size, scheme, right_interp);
    remote[v] = peer_buf[remote_off + v * buf_stride_var];
  }
  cs_sph_to_contra(remote, face, alpha, beta);
  if (eos_limiter) {
    own[IDN] = max(own[IDN], density_floor);
    own[IPR] = max(own[IPR], pressure_floor);
    remote[IDN] = max(remote[IDN], density_floor);
    remote[IPR] = max(remote[IPR], pressure_floor);
    for (int v = ICY; v < nvar; ++v) {
      own[v] = max(own[v], T(0));
      remote[v] = max(remote[v], T(0));
    }
  }
  bool lower_side = side == CS_SIDE_L || side == CS_SIDE_B;
  for (int v = 0; v < nvar; ++v) {
    wl[v] = lower_side ? remote[v] : own[v];
    wr[v] = lower_side ? own[v] : remote[v];
  }

  T el = 0., er = 0., gl = 0., gr = 0., cl = 0., cr = 0.;
  int ny = nvar - ICY;
  eos_side_quantities(wl, wr, ny, eos, gammad, inv_mu_ratio_m1, cv_ratio_m1,
                      u0, &el, &er, &gl, &gr, &cl, &cr);

  int flux_k = side <= CS_SIDE_R ? k : face_pos;
  int flux_j = side <= CS_SIDE_R ? face_pos : j;
  int flux_flat = flux_k * nc2 * nc1 + flux_j * nc1 + i;
  T* flux = side <= CS_SIDE_R ? flux2 + flux_flat : flux3 + flux_flat;
  if (solver == FusedRiemannSolver::LMARS) {
    lmars_impl(flux, wl, wr, el / wl[IDN], er / wr[IDN], gl, gr, dim, ny,
               stride_var);
  } else {
    hllc_impl(flux, wl, wr, el, er, gl, gr, cl, cr, dim, ny, stride_var);
  }
}

}  // namespace

void fused_recon_riemann_cuda(
    torch::Tensor w, torch::Tensor flux, int dim, FusedReconScheme recon_prim,
    FusedReconScheme recon_vel, FusedRiemannSolver solver, FusedEos eos,
    double gammad, double density_floor, double pressure_floor,
    bool eos_limiter, torch::Tensor inv_mu_ratio_m1,
    torch::Tensor cv_ratio_m1, torch::Tensor u0) {
  at::cuda::CUDAGuard device_guard(w.device());
  int nc3 = w.size(1);
  int nc2 = w.size(2);
  int nc1 = w.size(3);
  int nvar = w.size(0);
  int axis_size = dim == 3 ? nc1 : (dim == 2 ? nc2 : nc3);
  TORCH_CHECK(axis_size <= 1024,
              "dynamics.fused-recon-riemann shared-memory kernel requires "
              "the reconstructed dimension to fit in one CUDA block, but got ",
              axis_size);
  int threads = axis_size;
  int blocks = dim == 3 ? nc2 * nc3 : (dim == 2 ? nc1 * nc3 : nc1 * nc2);
  auto stream = at::cuda::getCurrentCUDAStream();

  AT_DISPATCH_FLOATING_TYPES(w.scalar_type(), "fused_recon_riemann_cuda", [&] {
    size_t shared = static_cast<size_t>(axis_size) * nvar * sizeof(scalar_t);
    fused_kernel<scalar_t><<<blocks, threads, shared, stream>>>(
        w.data_ptr<scalar_t>(), flux.data_ptr<scalar_t>(), nvar, nc3, nc2, nc1,
        dim, recon_prim, recon_vel, solver, eos, scalar_t(gammad),
        scalar_t(density_floor), scalar_t(pressure_floor), eos_limiter,
        inv_mu_ratio_m1.defined() ? inv_mu_ratio_m1.data_ptr<scalar_t>()
                                  : nullptr,
        cv_ratio_m1.defined() ? cv_ratio_m1.data_ptr<scalar_t>() : nullptr,
        u0.defined() ? u0.data_ptr<scalar_t>() : nullptr);
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void fused_cubed_sphere_exchange_cuda(
    torch::Tensor w, torch::Tensor flux2, torch::Tensor flux3,
    torch::Tensor symm_buffer, void** symm_buffer_ptrs_dev,
    uint32_t** symm_signal_pads_dev, int face, int symm_rank,
    int symm_world_size, torch::Tensor side_meta, torch::Tensor x2v,
    torch::Tensor x2f, torch::Tensor x3v, torch::Tensor x3f,
    FusedReconScheme recon_prim, FusedReconScheme recon_vel,
    FusedRiemannSolver solver, FusedEos eos, double gammad,
    double density_floor, double pressure_floor, bool eos_limiter,
    torch::Tensor inv_mu_ratio_m1, torch::Tensor cv_ratio_m1,
    torch::Tensor u0) {
  at::cuda::CUDAGuard device_guard(w.device());
  int nc3 = w.size(1);
  int nc2 = w.size(2);
  int nc1 = w.size(3);
  int nvar = w.size(0);
  int edge_len = std::max(nc2, nc3);
  int threads = edge_len;
  TORCH_CHECK(threads <= 1024,
              "dynamics.fused-recon-riemann cubed-sphere shared-memory "
              "exchange requires edge lines to fit in one CUDA block, but got ",
              threads);
  int blocks = 4 * edge_len * nc1;
  auto stream = at::cuda::getCurrentCUDAStream();

  AT_DISPATCH_FLOATING_TYPES(w.scalar_type(), "fused_cubed_sphere_exchange",
                             [&] {
    size_t shared = static_cast<size_t>(edge_len) * nvar * sizeof(scalar_t);
    cs_pack_kernel<scalar_t><<<blocks, threads, shared, stream>>>(
        w.data_ptr<scalar_t>(), symm_buffer.data_ptr<scalar_t>(),
        side_meta.data_ptr<int>(), nvar, nc3, nc2, nc1, face, recon_prim,
        recon_vel, scalar_t(density_floor), scalar_t(pressure_floor),
        eos_limiter, x2v.data_ptr<scalar_t>(), x2f.data_ptr<scalar_t>(),
        x3v.data_ptr<scalar_t>(), x3f.data_ptr<scalar_t>());
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  cs_sync_kernel<<<1, std::max(32, symm_world_size), 0, stream>>>(
      symm_signal_pads_dev, symm_rank, symm_world_size);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  AT_DISPATCH_FLOATING_TYPES(w.scalar_type(), "fused_cubed_sphere_flux", [&] {
    size_t shared = static_cast<size_t>(edge_len) * nvar * sizeof(scalar_t);
    cs_flux_kernel<scalar_t><<<blocks, threads, shared, stream>>>(
        w.data_ptr<scalar_t>(), flux2.defined() ? flux2.data_ptr<scalar_t>()
                                                : nullptr,
        flux3.defined() ? flux3.data_ptr<scalar_t>() : nullptr,
        symm_buffer_ptrs_dev, side_meta.data_ptr<int>(), nvar, nc3, nc2, nc1,
        face, recon_prim, recon_vel, solver, eos, scalar_t(gammad),
        scalar_t(density_floor), scalar_t(pressure_floor), eos_limiter,
        inv_mu_ratio_m1.defined() ? inv_mu_ratio_m1.data_ptr<scalar_t>()
                                  : nullptr,
        cv_ratio_m1.defined() ? cv_ratio_m1.data_ptr<scalar_t>() : nullptr,
        u0.defined() ? u0.data_ptr<scalar_t>() : nullptr,
        x2v.data_ptr<scalar_t>(), x2f.data_ptr<scalar_t>(),
        x3v.data_ptr<scalar_t>(), x3f.data_ptr<scalar_t>());
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
}  // namespace snap
