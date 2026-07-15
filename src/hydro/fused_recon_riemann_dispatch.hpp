#pragma once

// torch
#include <torch/torch.h>

// snap
#include <snap/snap.h>

namespace snap {

struct FusedPhysicsParams {
  FusedReconScheme recon_prim;
  FusedReconScheme recon_vel;
  FusedRiemannSolver solver;
  FusedEos eos;
  double gammad;
  double density_floor;
  double pressure_floor;
  bool eos_limiter;
  torch::Tensor inv_mu_ratio_m1;
  torch::Tensor cv_ratio_m1;
  torch::Tensor u0;
  int nvapor;
  int shallow_roe_dir_yz;
  //! honor reconstruct.scale in WENO schemes (CPU kernel only; the CUDA
  //! kernel does not read this field and keeps its scale=false behavior)
  bool recon_scale = false;
};

struct FusedX1RevisionParams {
  bool revise_lr;
  torch::Tensor dx1f;
  torch::Tensor rho_grav;
  // Per-face gates for the hydrostatic WALL revision at il/iu. On an
  // x1-decomposed grid (cubed nb1>1) an internal seam is not a physical wall,
  // so its wall swap must be suppressed (else it clobbers exchanged neighbor
  // data -- S2). Default true = both faces revised (single-block behavior,
  // and the CUDA path, unchanged). These do NOT gate rho_grav: the
  // hydrostatic divergence correction runs at every interior face regardless.
  bool revise_inner = true;
  bool revise_outer = true;
};

struct FusedMetricParams {
  bool cubed_sphere;
  int face;
  torch::Tensor x2v;
  torch::Tensor x2f;
  torch::Tensor x3v;
  torch::Tensor x3f;
};

struct FusedReconRiemannParams {
  int dim;
  FusedPhysicsParams physics;
  FusedX1RevisionParams x1_revision;
  FusedMetricParams metric;
  //! grid ghost width for the CPU kernel's ghost-line skip (0 = process all
  //! lines, the historical behavior; the CUDA kernel does not read this)
  int grid_nghost = 0;
};

void fused_recon_riemann_cuda(torch::Tensor w, torch::Tensor flux,
                              torch::Tensor face_pressure,
                              FusedReconRiemannParams const& params);

//! CPU port of the fused kernel (fused_recon_riemann_cpu.cpp). Cartesian
//! layouts only; params.metric.cubed_sphere must be false. `solid` (optional,
//! contiguous bool [nc3,nc2,nc1]) enables the internal-boundary face-state
//! revision (reflective wall), matching the staged pib->forward; the caller
//! must have applied mark_prim_solid_ to `w` first.
void fused_recon_riemann_cpu(torch::Tensor w, torch::Tensor flux,
                             torch::Tensor face_pressure, torch::Tensor solid,
                             FusedReconRiemannParams const& params);

// Multi-block-per-process cubed-sphere fused exchange, split into host-callable
// phases so the caller can coordinate the concurrent local-block threads:
// every local block packs its edge states into slice `local_block` of a shared
// per-process symmetric buffer, one designated block publishes cross-process
// visibility (sync), every local block computes and overwrites its cross-panel
// boundary flux (reading own + peer slices, peer selected by process rank), and
// one designated block closes the read epoch (release).
struct FusedCubedSpherePanelParams {
  torch::Tensor side_meta;
  int face;
  int local_block;
  torch::Tensor x2v;
  torch::Tensor x2f;
  torch::Tensor x3v;
  torch::Tensor x3f;
};

struct FusedCubedSpherePackParams {
  FusedCubedSpherePanelParams panel;
  FusedReconScheme recon_prim;
  FusedReconScheme recon_vel;
  FusedEos eos;
  double density_floor;
  double pressure_floor;
  bool eos_limiter;
};

void fused_cubed_sphere_pack_cuda(torch::Tensor w, torch::Tensor symm_buffer,
                                  FusedCubedSpherePackParams const& params);

void fused_cubed_sphere_sync_cuda(uint32_t** symm_signal_pads_dev,
                                  int symm_rank, int symm_world_size,
                                  torch::Device device);

struct FusedCubedSphereFluxParams {
  FusedCubedSpherePanelParams panel;
  FusedPhysicsParams physics;
};

void fused_cubed_sphere_flux_cuda(torch::Tensor w, torch::Tensor flux2,
                                  torch::Tensor flux3,
                                  torch::Tensor symm_buffer,
                                  void** symm_buffer_ptrs_dev,
                                  FusedCubedSphereFluxParams const& params);

// Process-level seam flux: one launch overwrites the cross-panel boundary flux
// for ALL local panels. flux2_ptrs_dev/flux3_ptrs_dev are device arrays of the
// per-panel flux tensor pointers (indexed by local block), faces is a device
// int array of each panel's face id, and side_meta_all is [bpp, 4, kStride].
// Coords are the shared equiangular angular grid (same for every panel).
struct FusedCubedSphereFluxAllPtrs {
  void** flux2;
  void** flux3;
  void** symm_buffer;
  void** x2v;
  void** x2f;
  void** x3v;
  void** x3f;
};

struct FusedCubedSphereFluxAllParams {
  torch::Tensor side_meta_all;
  torch::Tensor faces;
  int nvar;
  int nc3;
  int nc2;
  int nc1;
  int bpp;
  c10::ScalarType dtype;
  torch::Device device;
  FusedPhysicsParams physics;
};

void fused_cubed_sphere_flux_all_cuda(
    torch::Tensor symm_buffer, FusedCubedSphereFluxAllPtrs ptrs,
    FusedCubedSphereFluxAllParams const& params);

void fused_cubed_sphere_release_cuda(uint32_t** symm_signal_pads_dev,
                                     int symm_rank, int symm_world_size,
                                     torch::Device device);

}  // namespace snap
