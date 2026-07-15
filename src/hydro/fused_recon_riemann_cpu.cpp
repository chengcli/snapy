// CPU port of the fused reconstruction + Riemann kernel
// (fused_recon_riemann_dispatch.cu). One pass per line replaces the staged
// path's separate recon kernels, dummy-region fills, EOS temporaries and
// Riemann wrapper -- the ATen-dispatch fixed cost that dominates small
// blocks on CPU.
//
// Scope: cartesian layouts only. The cubed-sphere metric projection and the
// NVSHMEM seam exchange remain CUDA-only; callers must not pass
// metric.cubed_sphere = true.
//
// The physics helpers are the SAME headers the CUDA kernel uses
// (eos_side_quantities.cuh, hllc_impl.h, lmars_impl.h, shallow_roe_impl.h --
// all host-safe via DISPATCH_MACRO). The shared-line interpolation helpers
// below are transcribed from interp_impl.cuh, which is device-only; keep the
// two in sync.

// torch
#include <ATen/Dispatch.h>
#include <ATen/Parallel.h>
#include <torch/torch.h>

// C/C++
#include <cmath>
#include <vector>

// snap
#include <snap/snap.h>

#include "../eos/eos_side_quantities.cuh"
#include "../riemann/hllc_impl.h"
#include "../riemann/lmars_impl.h"
#include "../riemann/shallow_roe_impl.h"
#include "fused_dispatch_params.cuh"
#include "fused_recon_riemann_dispatch.hpp"

namespace snap {
namespace {

// ---------------------------------------------------------------------------
// transcribed from interp_impl.cuh (drop __device__); keep in sync
// ---------------------------------------------------------------------------
template <typename T>
inline T sqr_host(T x) {
  return x * x;
}

template <int N, typename T>
inline T vvdot_host(T const *v1, T const *v2) {
  T out = 0.;
  for (int i = 0; i < N; ++i) {
    out += v1[i] * v2[i];
  }
  return out;
}

template <typename T, int N>
T interp_line_poly_coeff(T const *line, T const *coeff, int v, int start,
                         int axis_size) {
  T out = 0.;
  for (int k = 0; k < N; ++k) {
    out += coeff[k] * line[v * axis_size + start + k];
  }
  return out;
}

template <typename T>
T interp_line_weno3_coeff(T const *line, T const *coeff, int v, int start,
                          int axis_size, bool scale) {
  T const *c1 = coeff;
  T const *c2 = c1 + 3;
  T const *c3 = c2 + 3;
  T const *c4 = c3 + 3;
  T const *src = line + v * axis_size + start;
  T vscale = scale ? (fabs(src[0]) + fabs(src[1]) + fabs(src[2])) / 3.0 : 1.0;

  if (vscale == 0.0) return 0.0;

  T phi[3];
  phi[0] = src[0] / vscale;
  phi[1] = src[1] / vscale;
  phi[2] = src[2] / vscale;

  T p0 = vvdot_host<3>(phi, c1);
  T p1 = vvdot_host<3>(phi, c2);

  T beta0 = sqr_host(vvdot_host<3>(phi, c3));
  T beta1 = sqr_host(vvdot_host<3>(phi, c4));

  T alpha0 = (1.0 / 3.0) / sqr_host(beta0 + 1e-6);
  T alpha1 = (2.0 / 3.0) / sqr_host(beta1 + 1e-6);

  return (alpha0 * p0 + alpha1 * p1) / (alpha0 + alpha1) * vscale;
}

template <typename T>
T interp_line_weno5_coeff(T const *line, T const *coeff, int v, int start,
                          int axis_size, bool scale) {
  T const *c1 = coeff;
  T const *c2 = c1 + 5;
  T const *c3 = c2 + 5;
  T const *c4 = c3 + 5;
  T const *c5 = c4 + 5;
  T const *c6 = c5 + 5;
  T const *c7 = c6 + 5;
  T const *c8 = c7 + 5;
  T const *c9 = c8 + 5;
  T const *src = line + v * axis_size + start;
  T vscale = scale ? (fabs(src[0]) + fabs(src[1]) + fabs(src[2]) +
                      fabs(src[3]) + fabs(src[4])) /
                         5.0
                   : 1.0;

  if (vscale == 0.0) return 0.0;

  T phi[5];
  for (int k = 0; k < 5; ++k) {
    phi[k] = src[k] / vscale;
  }

  T p0 = vvdot_host<5>(phi, c1);
  T p1 = vvdot_host<5>(phi, c2);
  T p2 = vvdot_host<5>(phi, c3);

  T beta0 = 13. / 12. * sqr_host(vvdot_host<5>(phi, c4)) +
            .25 * sqr_host(vvdot_host<5>(phi, c5));
  T beta1 = 13. / 12. * sqr_host(vvdot_host<5>(phi, c6)) +
            .25 * sqr_host(vvdot_host<5>(phi, c7));
  T beta2 = 13. / 12. * sqr_host(vvdot_host<5>(phi, c8)) +
            .25 * sqr_host(vvdot_host<5>(phi, c9));

  T alpha0 = .3 / sqr_host(beta0 + 1e-6);
  T alpha1 = .6 / sqr_host(beta1 + 1e-6);
  T alpha2 = .1 / sqr_host(beta2 + 1e-6);

  return vscale * (alpha0 * p0 + alpha1 * p1 + alpha2 * p2) /
         (alpha0 + alpha1 + alpha2);
}

//! precomputed coefficient tables (forward and row-reversed), built once per
//! kernel invocation instead of per (face, variable, side) -- identical
//! values. NOTE: like the CUDA kernel this path reconstructs with
//! scale=false and applies no shock limiter -- HydroImpl::forward refuses
//! the fused path when reconstruct.scale/shock are set.
template <typename T>
struct InterpTables {
  T cp3[2][3], cp5[2][5], w3[2][12], w5[2][45];

  InterpTables() {
    constexpr double cp3m[3] = {-1. / 3., 5. / 6., -1. / 6.};
    constexpr double cp5m[5] = {-1. / 20., 9. / 20., 47. / 60., -13. / 60.,
                                1. / 30.};
    constexpr double w3m[4][3] = {{1. / 2., 1. / 2., 0.},
                                  {0., 3. / 2., -1. / 2.},
                                  {1., -1., 0.},
                                  {0., 1., -1.}};
    constexpr double w5m[9][5] = {{-1. / 6., 5. / 6., 1. / 3., 0., 0.},
                                  {0., 1. / 3., 5. / 6., -1. / 6., 0.},
                                  {0., 0., 11. / 6., -7. / 6., 1. / 3.},
                                  {1., -2., 1., 0., 0.},
                                  {1., -4., 3., 0., 0.},
                                  {0., 1., -2., 1., 0.},
                                  {0., -1., 0., 1., 0.},
                                  {0., 0., 1., -2., 1.},
                                  {0., 0., 3., -4., 1.}};
    for (int k = 0; k < 3; ++k) {
      cp3[0][k] = cp3m[k];
      cp3[1][k] = cp3m[2 - k];
    }
    for (int k = 0; k < 5; ++k) {
      cp5[0][k] = cp5m[k];
      cp5[1][k] = cp5m[4 - k];
    }
    for (int r = 0; r < 4; ++r)
      for (int k = 0; k < 3; ++k) {
        w3[0][r * 3 + k] = w3m[r][k];
        w3[1][r * 3 + k] = w3m[r][2 - k];
      }
    for (int r = 0; r < 9; ++r)
      for (int k = 0; k < 5; ++k) {
        w5[0][r * 5 + k] = w5m[r][k];
        w5[1][r * 5 + k] = w5m[r][4 - k];
      }
  }
};

template <typename T>
T interp_line_fused(T const *line, int v, int start, int axis_size,
                    FusedReconScheme scheme, bool right,
                    InterpTables<T> const &tab, bool scale) {
  int s = right ? 1 : 0;
  if (scheme == FusedReconScheme::CP3) {
    return interp_line_poly_coeff<T, 3>(line, tab.cp3[s], v, start, axis_size);
  }
  if (scheme == FusedReconScheme::CP5) {
    return interp_line_poly_coeff<T, 5>(line, tab.cp5[s], v, start, axis_size);
  }
  if (scheme == FusedReconScheme::WENO3) {
    return interp_line_weno3_coeff(line, tab.w3[s], v, start, axis_size,
                                   scale);
  }
  return interp_line_weno5_coeff(line, tab.w5[s], v, start, axis_size,
                                 scale);
}

// ---------------------------------------------------------------------------
// one line = one CUDA block of the fused kernel; same statement order
// ---------------------------------------------------------------------------
template <typename T>
void fused_line_cpu(T const *w, T *flux,
                    DeviceFusedReconRiemannParams<T> const &params, int line,
                    InterpTables<T> const &tab, bool recon_scale,
                    bool revise_inner, bool revise_outer,
                    bool const *solid, T *line_buf, T *wl_buf, T *wr_buf,
                    T *lp_buf, T *rp_buf) {
  int nvar = params.nvar;
  int nc3 = params.nc3;
  int nc2 = params.nc2;
  int nc1 = params.nc1;
  int dim = params.dim;
  T *face_pressure = params.face_pressure;
  auto const &physics = params.physics;
  auto const &x1_revision = params.x1_revision;
  int ncells = nc1 * nc2 * nc3;
  int axis_size = dim == 3 ? nc1 : (dim == 2 ? nc2 : nc3);
  int stride_dim = dim == 3 ? 1 : (dim == 2 ? nc1 : nc1 * nc2);
  int stride_var = ncells;
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

  // gather the line (CUDA shared-memory load)
  for (int pos = 0; pos < axis_size; ++pos) {
    int flat = base + pos * stride_dim;
    for (int v = 0; v < nvar; ++v) {
      line_buf[v * axis_size + pos] = w[v * stride_var + flat];
    }
  }

  int nghost = (physics.recon_prim == FusedReconScheme::CP3 ||
                physics.recon_prim == FusedReconScheme::WENO3)
                   ? 2
                   : 3;
  int il = nghost;
  int iu = axis_size - nghost;
  int stencil = 2 * nghost - 1;
  bool revise = x1_revision.revise_lr && dim == 3;

  // pass 1: reconstruct L/R states at the valid faces (kernel pre-sync part;
  // recon at invalid positions is never read, so skip it)
  for (int pos = il; pos <= iu + 1; ++pos) {
    bool valid_face = true;
    int wl_start = std::min(std::max(pos - il, 0), axis_size - stencil);
    int wr_start = std::min(std::max(pos - (il - 1), 0), axis_size - stencil);
    T *wl_local = wl_buf + pos * nvar;
    T *wr_local = wr_buf + pos * nvar;
    for (int v = 0; v < nvar; ++v) {
      auto scheme =
          (v == IDN || v >= ICY) ? physics.recon_prim : physics.recon_vel;
      wl_local[v] = interp_line_fused(line_buf, v, wl_start, axis_size, scheme,
                                      /*right=*/true, tab, recon_scale);
      wr_local[v] = interp_line_fused(line_buf, v, wr_start, axis_size, scheme,
                                      /*right=*/false, tab, recon_scale);
    }

    if (revise && valid_face) {
      if (pos == il && revise_inner) {
        wl_local[IPR] = wr_local[IPR];
        wl_local[IDN] = wr_local[IDN];
      } else if (pos == iu && revise_outer) {
        // NOTE: deliberately pos == iu (the top INTERIOR face, matching the
        // staged _revise_x1outer_lr at pcoord->iu()+1). The CUDA kernel uses
        // pos == iu + 1, which is one past the last face the divergence
        // reads, so on GPU the outer revision is effectively skipped --
        // an upstream staged-vs-fused discrepancy.
        wr_local[IPR] = wl_local[IPR];
        wr_local[IDN] = wl_local[IDN];
      }
    }

    // solid internal-boundary revision, transcribed statement-for-statement
    // from InternalBoundaryImpl::forward (internal_boundary.cpp): a face
    // whose right cell is solid takes the mirrored left state with the
    // normal velocity sign-flipped (reflective wall), and vice versa; the
    // left-state revision reads the ALREADY-REVISED right state, matching
    // the staged op order. Placed exactly where the staged path calls
    // pib->forward: after the hydrostatic L/R revision, before anything
    // reads the face states (rho_grav below uses post-solid pressures).
    if (solid != nullptr) {
      int flat = base + pos * stride_dim;
      bool sr = solid[flat];  // cell at the face's right (staged solidl)
      // staged solidr = solid.roll(1) with edge fix: index 0 maps to itself
      bool sl = pos > 0 ? solid[flat - stride_dim] : sr;
      if (sr || sl) {
        int ivx = IPR - dim;
        for (int v = 0; v < nvar; ++v) {
          if (sr) wr_local[v] = wl_local[v];
          if (sl) wl_local[v] = wr_local[v];
        }
        if (sr) wr_local[ivx] = -wl_local[ivx];
        if (sl) wl_local[ivx] = -wr_local[ivx];
      }
    }

    // NOTE (semantics, inherited from the CUDA kernel): these floors apply
    // to reconstructed face states UNCONDITIONALLY when eos_limiter is on,
    // whereas the staged shock branch (reconstruct.cpp) early-returns with
    // no face clamping at all. With limiter:true + shock:true the two paths
    // can therefore diverge IF a WENO face value undershoots a floor; never
    // observed in validation (thermo bit-identical through 4000 cycles),
    // and clamping is the safer behavior, but it is a real difference.
    if (physics.eos_limiter && valid_face) {
      wl_local[IDN] = std::max(wl_local[IDN], physics.density_floor);
      wr_local[IDN] = std::max(wr_local[IDN], physics.density_floor);
      if (physics.eos != FusedEos::ShallowWater) {
        wl_local[IPR] = std::max(wl_local[IPR], physics.pressure_floor);
        wr_local[IPR] = std::max(wr_local[IPR], physics.pressure_floor);
        for (int v = ICY; v < nvar; ++v) {
          wl_local[v] = std::max(wl_local[v], T(0.));
          wr_local[v] = std::max(wr_local[v], T(0.));
        }
      }
    }

    if (revise && valid_face) {
      lp_buf[pos] = wl_local[IPR];
      rp_buf[pos] = wr_local[IPR];
    }
  }

  // pass 2: hydrostatic rho_grav (kernel post-sync part; same [il, iu) range)
  if (revise && x1_revision.rho_grav != nullptr) {
    for (int pos = il; pos < iu; ++pos) {
      int flat = base + pos * stride_dim;
      x1_revision.rho_grav[flat] =
          (lp_buf[pos + 1] - rp_buf[pos]) / x1_revision.dx1f[pos];
    }
  }

  // pass 3: fluxes
  for (int pos = 0; pos < axis_size; ++pos) {
    int flat = base + pos * stride_dim;
    bool valid_face = pos >= il && pos <= iu + 1;
    if (!valid_face) {
      for (int v = 0; v < nvar; ++v) flux[v * stride_var + flat] = 0.;
      if (face_pressure != nullptr) face_pressure[flat] = 0.;
      continue;
    }
    T *wl_local = wl_buf + pos * nvar;
    T *wr_local = wr_buf + pos * nvar;

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
      } else {
        hllc_impl(flux + flat, wl_local, wr_local, el, er, gl, gr, cl, cr, dim,
                  ny, /*stride_w=*/1, /*stride_f=*/stride_var,
                  face_pressure_out);
      }
    }
  }
}

}  // namespace

void fused_recon_riemann_cpu(torch::Tensor w, torch::Tensor flux,
                             torch::Tensor face_pressure, torch::Tensor solid,
                             FusedReconRiemannParams const &params) {
  TORCH_CHECK(w.device().is_cpu() && flux.device().is_cpu(),
              "fused_recon_riemann_cpu requires CPU tensors");
  TORCH_CHECK(!params.metric.cubed_sphere,
              "fused_recon_riemann_cpu supports cartesian layouts only");
  TORCH_CHECK(w.is_contiguous() && flux.is_contiguous(),
              "fused_recon_riemann_cpu requires contiguous tensors");
  TORCH_CHECK(!face_pressure.defined() ||
                  (face_pressure.device().is_cpu() &&
                   face_pressure.is_contiguous()),
              "fused_recon_riemann_cpu requires a contiguous CPU "
              "face_pressure tensor");
  TORCH_CHECK(params.physics.solver != FusedRiemannSolver::Roe,
              "fused_recon_riemann_cpu does not implement the roe solver");
  TORCH_CHECK(!solid.defined() ||
                  (solid.device().is_cpu() && solid.is_contiguous() &&
                   solid.scalar_type() == torch::kBool),
              "fused_recon_riemann_cpu requires a contiguous CPU bool solid "
              "tensor");

  int nc3 = w.size(1);
  int nc2 = w.size(2);
  int nc1 = w.size(3);
  int nvar = w.size(0);
  int dim = params.dim;
  int axis_size = dim == 3 ? nc1 : (dim == 2 ? nc2 : nc3);
  int64_t nlines = dim == 3 ? int64_t(nc2) * nc3
                            : (dim == 2 ? int64_t(nc1) * nc3
                                        : int64_t(nc1) * nc2);

  AT_DISPATCH_FLOATING_TYPES(w.scalar_type(), "fused_recon_riemann_cpu", [&] {
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
        {false, 0, nullptr, nullptr, nullptr, nullptr}};

    auto const *w_ptr = w.data_ptr<scalar_t>();
    auto *flux_ptr = flux.data_ptr<scalar_t>();
    bool const *solid_ptr = solid.defined() ? solid.data_ptr<bool>() : nullptr;
    InterpTables<scalar_t> const tab;

    // ghost-line skip: fluxes on lines outside the interior of the
    // non-swept dimensions are never consumed downstream (the divergence is
    // taken over the interior only and the flux buffers are zero-initialized
    // at registration, so skipped lines read as zero forever). grid_nghost
    // == 0 restores the historical process-every-line behavior.
    int const gng = params.grid_nghost;
    auto line_active = [&](int64_t line) -> bool {
      if (gng <= 0) return true;
      if (dim == 3) {
        int64_t j = line % nc2, k = line / nc2;
        return (nc2 == 1 || (j >= gng && j < nc2 - gng)) &&
               (nc3 == 1 || (k >= gng && k < nc3 - gng));
      }
      if (dim == 2) {
        int64_t i = line % nc1, k = line / nc1;
        return (nc1 == 1 || (i >= gng && i < nc1 - gng)) &&
               (nc3 == 1 || (k >= gng && k < nc3 - gng));
      }
      int64_t i = line % nc1, j = line / nc1;
      return (nc1 == 1 || (i >= gng && i < nc1 - gng)) &&
             (nc2 == 1 || (j >= gng && j < nc2 - gng));
    };

    at::parallel_for(0, nlines, /*grain_size=*/1, [&](int64_t b, int64_t e) {
      std::vector<scalar_t> line_buf(int64_t(nvar) * axis_size);
      std::vector<scalar_t> wl_buf(int64_t(nvar) * axis_size);
      std::vector<scalar_t> wr_buf(int64_t(nvar) * axis_size);
      std::vector<scalar_t> lp_buf(axis_size), rp_buf(axis_size);
      for (int64_t line = b; line < e; ++line) {
        if (!line_active(line)) continue;
        fused_line_cpu<scalar_t>(w_ptr, flux_ptr, device_params,
                                 static_cast<int>(line), tab,
                                 params.physics.recon_scale,
                                 params.x1_revision.revise_inner,
                                 params.x1_revision.revise_outer, solid_ptr,
                                 line_buf.data(), wl_buf.data(), wr_buf.data(),
                                 lp_buf.data(), rp_buf.data());
      }
    });
  });
}

}  // namespace snap
