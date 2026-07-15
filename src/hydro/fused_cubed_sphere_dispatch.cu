// torch
#include <ATen/Dispatch.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/csrc/distributed/c10d/symm_mem/CUDASymmetricMemory-inl.h>

// C/C++
#include <algorithm>

// snap
#include <snap/snap.h>

#include "../coord/gnomonic_equiangle.h"
#include "../eos/eos_side_quantities.cuh"
#include "../layout/cubed_sphere_constants.h"
#include "../recon/interp_impl.cuh"
#include "../riemann/hllc_impl.h"
#include "../riemann/lmars_impl.h"
#include "../riemann/roe_impl.h"
#include "../riemann/shallow_roe_impl.h"
#include "fused_dispatch_params.cuh"
#include "fused_recon_riemann_dispatch.hpp"

namespace snap {
namespace {

enum {
  CS_SIDE_L = 0,
  CS_SIDE_R = 1,
  CS_SIDE_B = 2,
  CS_SIDE_T = 3,
  CS_STATE_LEFT = ILT,
  CS_STATE_RIGHT = IRT,
  CS_NUM_STATES = 2,
  CS_NUM_SIDES = 4,
  CS_META_ENABLED = 0,
  CS_META_PEER_PROCESS = 1,
  CS_META_PEER_LOCAL_BLOCK = 2,
  CS_META_PEER_SIDE = 3,
  CS_META_REV = 4,
  CS_META_STRIDE = 5,
};

template <typename T>
__device__ void cs_coords(int side, int edge_pos, int i, int nc3, int nc2,
                          int nc1, int nghost, T const *x2v, T const *x2f,
                          T const *x3v, T const *x3f, T *alpha, T *beta, int *k,
                          int *j, int *face_pos) {
  (void)i;
  if (side == CS_SIDE_L || side == CS_SIDE_R) {
    *k = edge_pos;
    *j = side == CS_SIDE_L ? nghost : nc2 - nghost - 1;
    *face_pos = side == CS_SIDE_L ? nghost : nc2 - nghost;
    *alpha = x2f[side == CS_SIDE_L ? nghost : nc2 - nghost];
    *beta = x3v[edge_pos];
  } else {
    *k = side == CS_SIDE_B ? nghost : nc3 - nghost - 1;
    *j = edge_pos;
    *face_pos = side == CS_SIDE_B ? nghost : nc3 - nghost;
    *alpha = x2v[edge_pos];
    *beta = x3f[side == CS_SIDE_B ? nghost : nc3 - nghost];
  }
}

template <typename T>
__global__ void cs_pack_kernel(T const *w, T *buf,
                               DeviceCubedSpherePackParams<T> params) {
  int nvar = params.nvar;
  int nc3 = params.nc3;
  int nc2 = params.nc2;
  int nc1 = params.nc1;
  auto panel = params.panel;
  int edge_len = max(nc2, nc3);
  int line = blockIdx.x;
  int i = line % nc1;
  int edge_pos = (line / nc1) % edge_len;
  int side = line / (nc1 * edge_len);
  if (!panel.side_meta[side * CS_META_STRIDE + CS_META_ENABLED])
    return;
  if ((side <= CS_SIDE_R && edge_pos >= nc3) ||
      (side >= CS_SIDE_B && edge_pos >= nc2)) {
    return;
  }

  int dim = side <= CS_SIDE_R ? 2 : 1;
  int axis_size = dim == 2 ? nc2 : nc3;
  int stride_dim = dim == 2 ? nc1 : nc1 * nc2;
  int stride_var = nc1 * nc2 * nc3;
  int pos = threadIdx.x;
  int base = side <= CS_SIDE_R ? edge_pos * nc2 * nc1 + i : edge_pos * nc1 + i;

  extern __shared__ unsigned char memory[];
  T *smem = reinterpret_cast<T *>(memory);
  if (pos < axis_size) {
    int in = base + pos * stride_dim;
    for (int v = 0; v < nvar; ++v) {
      smem[v * axis_size + pos] = w[v * stride_var + in];
    }
  }
  __syncthreads();

  int nghost = (params.recon_prim == FusedReconScheme::CP3 ||
                params.recon_prim == FusedReconScheme::WENO3)
                   ? 2
                   : 3;
  int k, j, face_pos;
  T alpha, beta;
  cs_coords(side, edge_pos, i, nc3, nc2, nc1, nghost, panel.x2v, panel.x2f,
            panel.x3v, panel.x3f, &alpha, &beta, &k, &j, &face_pos);
  if (pos != face_pos)
    return;

  int il = nghost;
  int left_start = face_pos - il;
  int right_start = face_pos - (il - 1);
  T left[64], right[64];
  for (int v = 0; v < nvar; ++v) {
    auto scheme = (v == IDN || v >= ICY) ? params.recon_prim : params.recon_vel;
    left[v] = interp_shared_fused_impl(smem, v, left_start, axis_size, scheme,
                                       /*right=*/true, params.recon_scale);
    right[v] = interp_shared_fused_impl(smem, v, right_start, axis_size, scheme,
                                        /*right=*/false, params.recon_scale);
  }
  if (params.eos_limiter) {
    left[IDN] = max(left[IDN], params.density_floor);
    right[IDN] = max(right[IDN], params.density_floor);
    if (params.eos != FusedEos::ShallowWater) {
      left[IPR] = max(left[IPR], params.pressure_floor);
      right[IPR] = max(right[IPR], params.pressure_floor);
      for (int v = ICY; v < nvar; ++v) {
        left[v] = max(left[v], T(0));
        right[v] = max(right[v], T(0));
      }
    }
  }
  gnomonic_contra_to_sph(left, panel.face, alpha, beta);
  gnomonic_contra_to_sph(right, panel.face, alpha, beta);

  int buf_stride_var = edge_len * nc1;
  int block_stride = CS_NUM_SIDES * CS_NUM_STATES * nvar * edge_len * nc1;
  int block_base = panel.local_block * block_stride;
  int left_out =
      block_base +
      (((side * CS_NUM_STATES + CS_STATE_LEFT) * nvar) * edge_len + edge_pos) *
          nc1 +
      i;
  int right_out =
      block_base +
      (((side * CS_NUM_STATES + CS_STATE_RIGHT) * nvar) * edge_len + edge_pos) *
          nc1 +
      i;
  for (int v = 0; v < nvar; ++v) {
    buf[left_out + v * buf_stride_var] = left[v];
    buf[right_out + v * buf_stride_var] = right[v];
  }
}

__global__ void cs_sync_kernel(uint32_t **signal_pads, int rank,
                               int world_size) {
  c10d::symmetric_memory::sync_remote_blocks<false, true>(signal_pads, rank,
                                                          world_size);
}

__global__ void cs_release_reads_kernel(uint32_t **signal_pads, int rank,
                                        int world_size) {
  if (blockIdx.x != 1)
    return;
  c10d::symmetric_memory::sync_remote_blocks<true, false>(signal_pads, rank,
                                                          world_size);
}

template <typename T>
__device__ void cs_flux_common(DeviceCubedSphereFluxParams<T> params, int i,
                               int edge_pos, int side) {
  T const *buf = params.buf;
  T *flux2 = params.flux2;
  T *flux3 = params.flux3;
  void **buf_ptrs = params.buf_ptrs;
  int nvar = params.nvar;
  int nc3 = params.nc3;
  int nc2 = params.nc2;
  int nc1 = params.nc1;
  auto panel = params.panel;
  auto physics = params.physics;
  int const *side_meta = panel.side_meta;
  if (!side_meta[side * CS_META_STRIDE + CS_META_ENABLED])
    return;
  if ((side <= CS_SIDE_R && edge_pos >= nc3) ||
      (side >= CS_SIDE_B && edge_pos >= nc2)) {
    return;
  }

  int peer_process = side_meta[side * CS_META_STRIDE + CS_META_PEER_PROCESS];
  int peer_local_block =
      side_meta[side * CS_META_STRIDE + CS_META_PEER_LOCAL_BLOCK];
  int peer_side = side_meta[side * CS_META_STRIDE + CS_META_PEER_SIDE];
  int rev = side_meta[side * CS_META_STRIDE + CS_META_REV];
  int peer_edge =
      rev ? ((side <= CS_SIDE_R ? nc3 : nc2) - 1 - edge_pos) : edge_pos;
  auto peer_buf = static_cast<T const *>(buf_ptrs[peer_process]);
  int edge_len = max(nc2, nc3);
  int buf_stride_var = edge_len * nc1;
  int block_stride = CS_NUM_SIDES * CS_NUM_STATES * nvar * edge_len * nc1;
  int local_base = panel.local_block * block_stride;
  int peer_base = peer_local_block * block_stride;
  int remote_left_off =
      peer_base +
      (((peer_side * CS_NUM_STATES + CS_STATE_LEFT) * nvar) * edge_len +
       peer_edge) *
          nc1 +
      i;
  int remote_right_off =
      peer_base +
      (((peer_side * CS_NUM_STATES + CS_STATE_RIGHT) * nvar) * edge_len +
       peer_edge) *
          nc1 +
      i;
  int local_left_off =
      local_base +
      (((side * CS_NUM_STATES + CS_STATE_LEFT) * nvar) * edge_len + edge_pos) *
          nc1 +
      i;
  int local_right_off =
      local_base +
      (((side * CS_NUM_STATES + CS_STATE_RIGHT) * nvar) * edge_len + edge_pos) *
          nc1 +
      i;

  int dim = side <= CS_SIDE_R ? 2 : 1;
  int nghost = (physics.recon_prim == FusedReconScheme::CP3 ||
                physics.recon_prim == FusedReconScheme::WENO3)
                   ? 2
                   : 3;
  int k, j, face_pos;
  T alpha, beta;
  cs_coords(side, edge_pos, i, nc3, nc2, nc1, nghost, panel.x2v, panel.x2f,
            panel.x3v, panel.x3f, &alpha, &beta, &k, &j, &face_pos);
  if (threadIdx.x != face_pos)
    return;

  T local_left[64], local_right[64], remote_left[64], remote_right[64], wl[64],
      wr[64];
  for (int v = 0; v < nvar; ++v) {
    local_left[v] = buf[local_left_off + v * buf_stride_var];
    local_right[v] = buf[local_right_off + v * buf_stride_var];
    remote_left[v] = peer_buf[remote_left_off + v * buf_stride_var];
    remote_right[v] = peer_buf[remote_right_off + v * buf_stride_var];
  }
  bool lower_side = side == CS_SIDE_L || side == CS_SIDE_B;
  bool peer_lower_side = peer_side == CS_SIDE_L || peer_side == CS_SIDE_B;
  for (int v = 0; v < nvar; ++v) {
    T remote = peer_lower_side ? remote_right[v] : remote_left[v];
    wl[v] = lower_side ? remote : local_left[v];
    wr[v] = lower_side ? local_right[v] : remote;
  }
  gnomonic_sph_to_contra(wl, panel.face, alpha, beta);
  gnomonic_sph_to_contra(wr, panel.face, alpha, beta);
  if (physics.eos_limiter) {
    wl[IDN] = max(wl[IDN], physics.density_floor);
    wr[IDN] = max(wr[IDN], physics.density_floor);
    if (physics.eos != FusedEos::ShallowWater) {
      wl[IPR] = max(wl[IPR], physics.pressure_floor);
      wr[IPR] = max(wr[IPR], physics.pressure_floor);
      for (int v = ICY; v < nvar; ++v) {
        wl[v] = max(wl[v], T(0));
        wr[v] = max(wr[v], T(0));
      }
    }
  }
  T wl_density = wl[IDN];
  T wr_density = wr[IDN];
  gnomonic_prim2local(wl, dim, alpha, beta);
  gnomonic_prim2local(wr, dim, alpha, beta);
  wl[IDN] = wl_density;
  wr[IDN] = wr_density;

  int flux_k = side <= CS_SIDE_R ? k : face_pos;
  int flux_j = side <= CS_SIDE_R ? face_pos : j;
  int flux_flat = flux_k * nc2 * nc1 + flux_j * nc1 + i;
  int stride_var = nc1 * nc2 * nc3;
  T *flux = side <= CS_SIDE_R ? flux2 + flux_flat : flux3 + flux_flat;
  if (physics.solver == FusedRiemannSolver::ShallowRoe) {
    shallow_roe_impl(flux, wl, wr, dim, physics.shallow_roe_dir_yz,
                     /*stride_w=*/1, /*stride_f=*/stride_var);
    gnomonic_flux2global(flux, dim, alpha, beta, stride_var);
    return;
  }

  T el = 0., er = 0., gl = 0., gr = 0., cl = 0., cr = 0.;
  int ny = nvar - ICY;
  eos_side_quantities(wl, wr, ny, physics.nvapor, physics.eos, physics.gammad,
                      physics.inv_mu_ratio_m1, physics.cv_ratio_m1, physics.u0,
                      &el, &er, &gl, &gr, &cl, &cr);

  if (physics.solver == FusedRiemannSolver::LMARS) {
    lmars_impl(flux, wl, wr, el / wl[IDN], er / wr[IDN], gl, gr, dim, ny,
               /*stride_w=*/1, /*stride_f=*/stride_var);
  } else if (physics.solver == FusedRiemannSolver::HLLC) {
    hllc_impl(flux, wl, wr, el, er, gl, gr, cl, cr, dim, ny, /*stride_w=*/1,
              /*stride_f=*/stride_var);
  } else {
    roe_impl(flux, wl, wr, el, er, gl, gr, cl, cr, dim, ny, physics.eos,
             physics.nvapor, physics.gammad, physics.inv_mu_ratio_m1,
             physics.cv_ratio_m1, physics.u0, /*stride_w=*/1,
             /*stride_f=*/stride_var);
  }
  gnomonic_flux2global(flux, dim, alpha, beta, stride_var);
}

template <typename T>
__global__ void cs_flux_kernel(DeviceCubedSphereFluxParams<T> params) {
  (void)params.physics.recon_vel;
  int edge_len = max(params.nc2, params.nc3);
  int line = blockIdx.x;
  int i = line % params.nc1;
  int edge_pos = (line / params.nc1) % edge_len;
  int side = line / (params.nc1 * edge_len);
  cs_flux_common(params, i, edge_pos, side);
}

template <typename T>
__global__ void cs_flux_all_kernel(DeviceCubedSphereFluxParams<T> base_params,
                                   void **flux2_ptrs, void **flux3_ptrs,
                                   int const *side_meta_all, int const *faces,
                                   void **x2v_ptrs, void **x2f_ptrs,
                                   void **x3v_ptrs, void **x3f_ptrs) {
  (void)base_params.physics.recon_vel;
  int edge_len = max(base_params.nc2, base_params.nc3);
  int lines_per_panel = CS_NUM_SIDES * edge_len * base_params.nc1;
  int panel = blockIdx.x / lines_per_panel;
  int line = blockIdx.x % lines_per_panel;
  int i = line % base_params.nc1;
  int edge_pos = (line / base_params.nc1) % edge_len;
  int side = line / (base_params.nc1 * edge_len);
  auto params = base_params;
  params.flux2 = static_cast<T *>(flux2_ptrs[panel]);
  params.flux3 = static_cast<T *>(flux3_ptrs[panel]);
  params.panel.side_meta =
      side_meta_all + panel * CS_NUM_SIDES * CS_META_STRIDE;
  params.panel.face = faces[panel];
  params.panel.local_block = panel;
  params.panel.x2v = static_cast<T const *>(x2v_ptrs[panel]);
  params.panel.x2f = static_cast<T const *>(x2f_ptrs[panel]);
  params.panel.x3v = static_cast<T const *>(x3v_ptrs[panel]);
  params.panel.x3f = static_cast<T const *>(x3f_ptrs[panel]);
  cs_flux_common(params, i, edge_pos, side);
}

} // namespace

void fused_cubed_sphere_pack_cuda(torch::Tensor w, torch::Tensor symm_buffer,
                                  FusedCubedSpherePackParams const &params) {
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

  AT_DISPATCH_FLOATING_TYPES(w.scalar_type(), "fused_cubed_sphere_pack", [&] {
    size_t shared = static_cast<size_t>(edge_len) * nvar * sizeof(scalar_t);
    DeviceCubedSpherePackParams<scalar_t> device_params{
        nvar,
        nc3,
        nc2,
        nc1,
        make_device_panel<scalar_t>(params.panel),
        params.recon_prim,
        params.recon_vel,
        params.recon_scale,
        params.eos,
        scalar_t(params.density_floor),
        scalar_t(params.pressure_floor),
        params.eos_limiter};
    cs_pack_kernel<scalar_t><<<blocks, threads, shared, stream>>>(
        w.data_ptr<scalar_t>(), symm_buffer.data_ptr<scalar_t>(),
        device_params);
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void fused_cubed_sphere_sync_cuda(uint32_t **symm_signal_pads_dev,
                                  int symm_rank, int symm_world_size,
                                  torch::Device device) {
  at::cuda::CUDAGuard device_guard(device);
  auto stream = at::cuda::getCurrentCUDAStream();
  cs_sync_kernel<<<1, std::max(32, symm_world_size), 0, stream>>>(
      symm_signal_pads_dev, symm_rank, symm_world_size);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void fused_cubed_sphere_flux_cuda(torch::Tensor w, torch::Tensor flux2,
                                  torch::Tensor flux3,
                                  torch::Tensor symm_buffer,
                                  void **symm_buffer_ptrs_dev,
                                  FusedCubedSphereFluxParams const &params) {
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

  AT_DISPATCH_FLOATING_TYPES(w.scalar_type(), "fused_cubed_sphere_flux", [&] {
    DeviceCubedSphereFluxParams<scalar_t> device_params{
        symm_buffer.data_ptr<scalar_t>(),
        flux2.defined() ? flux2.data_ptr<scalar_t>() : nullptr,
        flux3.defined() ? flux3.data_ptr<scalar_t>() : nullptr,
        symm_buffer_ptrs_dev,
        nvar,
        nc3,
        nc2,
        nc1,
        make_device_panel<scalar_t>(params.panel),
        make_device_physics<scalar_t>(params.physics)};
    cs_flux_kernel<scalar_t><<<blocks, threads, 0, stream>>>(device_params);
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void fused_cubed_sphere_flux_all_cuda(
    torch::Tensor symm_buffer, FusedCubedSphereFluxAllPtrs ptrs,
    FusedCubedSphereFluxAllParams const &params) {
  at::cuda::CUDAGuard device_guard(params.device);
  int edge_len = std::max(params.nc2, params.nc3);
  int threads = edge_len;
  TORCH_CHECK(threads <= 1024,
              "dynamics.fused-recon-riemann cubed-sphere shared-memory "
              "exchange requires edge lines to fit in one CUDA block, but got ",
              threads);
  int blocks = params.bpp * 4 * edge_len * params.nc1;
  auto stream = at::cuda::getCurrentCUDAStream();

  AT_DISPATCH_FLOATING_TYPES(params.dtype, "fused_cubed_sphere_flux_all", [&] {
    DeviceCubedSphereFluxParams<scalar_t> device_params{
        symm_buffer.data_ptr<scalar_t>(),
        nullptr,
        nullptr,
        ptrs.symm_buffer,
        params.nvar,
        params.nc3,
        params.nc2,
        params.nc1,
        {nullptr, 0, 0, nullptr, nullptr, nullptr, nullptr},
        make_device_physics<scalar_t>(params.physics)};
    cs_flux_all_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
        device_params, ptrs.flux2, ptrs.flux3,
        params.side_meta_all.data_ptr<int>(), params.faces.data_ptr<int>(),
        ptrs.x2v, ptrs.x2f, ptrs.x3v, ptrs.x3f);
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void fused_cubed_sphere_release_cuda(uint32_t **symm_signal_pads_dev,
                                     int symm_rank, int symm_world_size,
                                     torch::Device device) {
  at::cuda::CUDAGuard device_guard(device);
  auto stream = at::cuda::getCurrentCUDAStream();
  cs_release_reads_kernel<<<2, std::max(32, symm_world_size), 0, stream>>>(
      symm_signal_pads_dev, symm_rank, symm_world_size);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

} // namespace snap
