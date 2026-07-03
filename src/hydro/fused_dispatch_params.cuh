#pragma once

#include "fused_recon_riemann_dispatch.hpp"

namespace snap {

template <typename T>
struct DeviceFusedPhysicsParams {
  FusedReconScheme recon_prim;
  FusedReconScheme recon_vel;
  FusedRiemannSolver solver;
  FusedEos eos;
  T gammad;
  T density_floor;
  T pressure_floor;
  bool eos_limiter;
  T const* inv_mu_ratio_m1;
  T const* cv_ratio_m1;
  T const* u0;
  int nvapor;
  int shallow_roe_dir_yz;
};

template <typename T>
struct DeviceFusedX1RevisionParams {
  bool revise_lr;
  T const* dx1f;
  T* rho_grav;
};

template <typename T>
struct DeviceFusedMetricParams {
  bool cubed_sphere;
  int face;
  T const* x2v;
  T const* x2f;
  T const* x3v;
  T const* x3f;
};

template <typename T>
struct DeviceFusedReconRiemannParams {
  int nvar;
  int nc3;
  int nc2;
  int nc1;
  int dim;
  DeviceFusedPhysicsParams<T> physics;
  DeviceFusedX1RevisionParams<T> x1_revision;
  DeviceFusedMetricParams<T> metric;
};

template <typename T>
struct DeviceCubedSpherePanelParams {
  int const* side_meta;
  int face;
  int local_block;
  T const* x2v;
  T const* x2f;
  T const* x3v;
  T const* x3f;
};

template <typename T>
struct DeviceCubedSpherePackParams {
  int nvar;
  int nc3;
  int nc2;
  int nc1;
  DeviceCubedSpherePanelParams<T> panel;
  FusedReconScheme recon_prim;
  FusedReconScheme recon_vel;
  FusedEos eos;
  T density_floor;
  T pressure_floor;
  bool eos_limiter;
};

template <typename T>
struct DeviceCubedSphereFluxParams {
  T const* buf;
  T* flux2;
  T* flux3;
  void** buf_ptrs;
  int nvar;
  int nc3;
  int nc2;
  int nc1;
  DeviceCubedSpherePanelParams<T> panel;
  DeviceFusedPhysicsParams<T> physics;
};

template <typename T>
inline DeviceFusedPhysicsParams<T> make_device_physics(
    FusedPhysicsParams const& p) {
  return {p.recon_prim,
          p.recon_vel,
          p.solver,
          p.eos,
          T(p.gammad),
          T(p.density_floor),
          T(p.pressure_floor),
          p.eos_limiter,
          p.inv_mu_ratio_m1.defined() ? p.inv_mu_ratio_m1.data_ptr<T>()
                                      : nullptr,
          p.cv_ratio_m1.defined() ? p.cv_ratio_m1.data_ptr<T>() : nullptr,
          p.u0.defined() ? p.u0.data_ptr<T>() : nullptr,
          p.nvapor,
          p.shallow_roe_dir_yz};
}

template <typename T>
inline DeviceCubedSpherePanelParams<T> make_device_panel(
    FusedCubedSpherePanelParams const& p) {
  return {p.side_meta.data_ptr<int>(),
          p.face,
          p.local_block,
          p.x2v.data_ptr<T>(),
          p.x2f.data_ptr<T>(),
          p.x3v.data_ptr<T>(),
          p.x3f.data_ptr<T>()};
}

}  // namespace snap
