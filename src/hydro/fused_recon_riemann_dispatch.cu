// torch
#include <ATen/Dispatch.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

// snap
#include <snap/snap.h>

#include "../coord/gnomonic_equiangle.h"
#include "../eos/eos_side_quantities.cuh"
#include "../recon/interp_impl.cuh"
#include "../riemann/hllc_impl.h"
#include "../riemann/lmars_impl.h"
#include "../riemann/roe_impl.h"
#include "../riemann/shallow_roe_impl.h"
#include "fused_dispatch_params.cuh"
#include "fused_recon_riemann_dispatch.hpp"

namespace snap {
namespace {

template <typename T>
__global__ void fused_kernel(T const *w, T *flux,
                             DeviceFusedReconRiemannParams<T> params) {
  int nvar = params.nvar;
  int nc3 = params.nc3;
  int nc2 = params.nc2;
  int nc1 = params.nc1;
  int dim = params.dim;
  T *face_pressure = params.face_pressure;
  auto physics = params.physics;
  auto x1_revision = params.x1_revision;
  auto metric = params.metric;
  int ncells = nc1 * nc2 * nc3;
  int axis_size = dim == 3 ? nc1 : (dim == 2 ? nc2 : nc3);
  int stride_dim = dim == 3 ? 1 : (dim == 2 ? nc1 : nc1 * nc2);
  int stride_var = ncells;
  int pos = threadIdx.x;
  int line = blockIdx.x;
  int base = 0;

  int i = 0;
  int j = 0;
  int k = 0;
  if (dim == 3) {
    j = line % nc2;
    k = line / nc2;
    base = k * nc2 * nc1 + j * nc1;
  } else if (dim == 2) {
    i = line % nc1;
    k = line / nc1;
    base = k * nc2 * nc1 + i;
  } else {
    i = line % nc1;
    j = line / nc1;
    base = j * nc1 + i;
  }

  extern __shared__ unsigned char memory[];
  T *smem = reinterpret_cast<T *>(memory);
  T *left_pressure = smem + nvar * axis_size;
  T *right_pressure = left_pressure + axis_size;
  if (pos < axis_size) {
    int flat = base + pos * stride_dim;
    for (int v = 0; v < nvar; ++v) {
      smem[v * axis_size + pos] = w[v * stride_var + flat];
    }
  }
  __syncthreads();

  if (pos >= axis_size)
    return;
  int flat = base + pos * stride_dim;
  int nghost = (physics.recon_prim == FusedReconScheme::CP3 ||
                physics.recon_prim == FusedReconScheme::WENO3)
                   ? 2
                   : 3;
  int il = nghost;
  int iu = axis_size - nghost;
  int stencil = 2 * nghost - 1;

  bool valid_face = pos >= il && pos <= iu + 1;

  int wl_start = min(max(pos - il, 0), axis_size - stencil);
  int wr_start = min(max(pos - (il - 1), 0), axis_size - stencil);
  T wl_local[64], wr_local[64];
  for (int v = 0; v < nvar; ++v) {
    auto scheme =
        (v == IDN || v >= ICY) ? physics.recon_prim : physics.recon_vel;
    wl_local[v] = interp_shared_fused_impl(smem, v, wl_start, axis_size, scheme,
                                           /*right=*/true, physics.recon_scale);
    wr_local[v] =
        interp_shared_fused_impl(smem, v, wr_start, axis_size, scheme,
                                 /*right=*/false, physics.recon_scale);
  }

  if (x1_revision.revise_lr && dim == 3 && valid_face) {
    if (pos == il) {
      wl_local[IPR] = wr_local[IPR];
      wl_local[IDN] = wr_local[IDN];
    } else if (pos == iu) {
      wr_local[IPR] = wl_local[IPR];
      wr_local[IDN] = wl_local[IDN];
    }
  }

  if (physics.eos_limiter && valid_face) {
    wl_local[IDN] = max(wl_local[IDN], physics.density_floor);
    wr_local[IDN] = max(wr_local[IDN], physics.density_floor);
    if (physics.eos != FusedEos::ShallowWater) {
      wl_local[IPR] = max(wl_local[IPR], physics.pressure_floor);
      wr_local[IPR] = max(wr_local[IPR], physics.pressure_floor);
      for (int v = ICY; v < nvar; ++v) {
        wl_local[v] = max(wl_local[v], T(0.));
        wr_local[v] = max(wr_local[v], T(0.));
      }
    }
  }

  if (x1_revision.revise_lr && dim == 3 && valid_face) {
    left_pressure[pos] = wl_local[IPR];
    right_pressure[pos] = wr_local[IPR];
  }
  __syncthreads();
  if (x1_revision.revise_lr && dim == 3 && x1_revision.rho_grav != nullptr &&
      pos >= il && pos < iu) {
    x1_revision.rho_grav[flat] =
        (left_pressure[pos + 1] - right_pressure[pos]) / x1_revision.dx1f[pos];
  }

  if (!valid_face) {
    for (int v = 0; v < nvar; ++v)
      flux[v * stride_var + flat] = 0.;
    if (face_pressure != nullptr)
      face_pressure[flat] = 0.;
    return;
  }

  T alpha = 0;
  T beta = 0;
  bool use_cubed_metric = metric.cubed_sphere && dim != 3;
  if (use_cubed_metric) {
    if (dim == 2) {
      alpha = metric.x2f[pos];
      beta = metric.x3v[k];
    } else {
      alpha = metric.x2v[j];
      beta = metric.x3f[pos];
    }
    T wl_density = wl_local[IDN];
    T wr_density = wr_local[IDN];
    gnomonic_prim2local(wl_local, dim, alpha, beta);
    gnomonic_prim2local(wr_local, dim, alpha, beta);
    wl_local[IDN] = wl_density;
    wr_local[IDN] = wr_density;
  }

  if (physics.solver == FusedRiemannSolver::ShallowRoe) {
    shallow_roe_impl(flux + flat, wl_local, wr_local, dim,
                     physics.shallow_roe_dir_yz, /*stride_w=*/1,
                     /*stride_f=*/stride_var);
  } else {
    T *face_pressure_out =
        face_pressure != nullptr ? face_pressure + flat : nullptr;
    T el = 0., er = 0., gl = 0., gr = 0., cl = 0., cr = 0.;
    int ny = physics.eos == FusedEos::ShallowWater ? 0 : nvar - ICY;
    eos_side_quantities(wl_local, wr_local, ny, physics.nvapor, physics.eos,
                        physics.gammad, physics.inv_mu_ratio_m1,
                        physics.cv_ratio_m1, physics.u0, &el, &er, &gl, &gr,
                        &cl, &cr);

    if (physics.solver == FusedRiemannSolver::LMARS) {
      lmars_impl(flux + flat, wl_local, wr_local, el / wl_local[IDN],
                 er / wr_local[IDN], gl, gr, dim, ny, /*stride_w=*/1,
                 /*stride_f=*/stride_var, face_pressure_out);
    } else if (physics.solver == FusedRiemannSolver::HLLC) {
      hllc_impl(flux + flat, wl_local, wr_local, el, er, gl, gr, cl, cr, dim,
                ny, /*stride_w=*/1, /*stride_f=*/stride_var, face_pressure_out);
    } else {
      roe_impl(flux + flat, wl_local, wr_local, el, er, gl, gr, cl, cr, dim, ny,
               physics.eos, physics.nvapor, physics.gammad,
               physics.inv_mu_ratio_m1, physics.cv_ratio_m1, physics.u0,
               /*stride_w=*/1, /*stride_f=*/stride_var, face_pressure_out);
    }
  }
  if (use_cubed_metric) {
    gnomonic_flux2global(flux + flat, dim, alpha, beta, stride_var);
  }
}

} // namespace

void fused_recon_riemann_cuda(torch::Tensor w, torch::Tensor flux,
                              torch::Tensor face_pressure,
                              FusedReconRiemannParams const &params) {
  at::cuda::CUDAGuard device_guard(w.device());
  int nc3 = w.size(1);
  int nc2 = w.size(2);
  int nc1 = w.size(3);
  int nvar = w.size(0);
  int dim = params.dim;
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
    if (params.x1_revision.revise_lr && dim == 3) {
      shared += static_cast<size_t>(2) * axis_size * sizeof(scalar_t);
    }
    DeviceFusedReconRiemannParams<scalar_t> device_params{
        nvar,
        nc3,
        nc2,
        nc1,
        dim,
        face_pressure.defined() ? face_pressure.data_ptr<scalar_t>() : nullptr,
        make_device_physics<scalar_t>(params.physics),
        {params.x1_revision.revise_lr,
         params.x1_revision.dx1f.defined()
             ? params.x1_revision.dx1f.data_ptr<scalar_t>()
             : nullptr,
         params.x1_revision.rho_grav.defined()
             ? params.x1_revision.rho_grav.data_ptr<scalar_t>()
             : nullptr},
        {params.metric.cubed_sphere, params.metric.face,
         params.metric.x2v.defined() ? params.metric.x2v.data_ptr<scalar_t>()
                                     : nullptr,
         params.metric.x2f.defined() ? params.metric.x2f.data_ptr<scalar_t>()
                                     : nullptr,
         params.metric.x3v.defined() ? params.metric.x3v.data_ptr<scalar_t>()
                                     : nullptr,
         params.metric.x3f.defined() ? params.metric.x3f.data_ptr<scalar_t>()
                                     : nullptr}};
    fused_kernel<scalar_t><<<blocks, threads, shared, stream>>>(
        w.data_ptr<scalar_t>(), flux.data_ptr<scalar_t>(), device_params);
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

} // namespace snap
