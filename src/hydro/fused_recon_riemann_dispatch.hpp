#pragma once

// torch
#include <torch/torch.h>

// snap
#include <snap/snap.h>

namespace snap {

void fused_recon_riemann_cuda(
    torch::Tensor w, torch::Tensor flux, int dim, FusedReconScheme recon_prim,
    FusedReconScheme recon_vel, FusedRiemannSolver solver, FusedEos eos,
    double gammad, double density_floor, double pressure_floor,
    bool eos_limiter, torch::Tensor inv_mu_ratio_m1, torch::Tensor cv_ratio_m1,
    torch::Tensor u0, int shallow_roe_dir_yz, FusedPrimitiveProjector projector,
    torch::Tensor psf, torch::Tensor dx1f, double gas_constant,
    torch::Tensor rho_grav, bool cubed_sphere, int face, torch::Tensor x2v,
    torch::Tensor x2f, torch::Tensor x3v, torch::Tensor x3f);

// Multi-block-per-process cubed-sphere fused exchange, split into host-callable
// phases so the caller can coordinate the concurrent local-block threads:
// every local block packs its edge states into slice `local_block` of a shared
// per-process symmetric buffer, one designated block publishes cross-process
// visibility (sync), every local block computes and overwrites its cross-panel
// boundary flux (reading own + peer slices, peer selected by process rank), and
// one designated block closes the read epoch (release).
void fused_cubed_sphere_pack_cuda(
    torch::Tensor w, torch::Tensor symm_buffer, torch::Tensor side_meta,
    int face, int local_block, torch::Tensor x2v, torch::Tensor x2f,
    torch::Tensor x3v, torch::Tensor x3f, FusedReconScheme recon_prim,
    FusedReconScheme recon_vel, FusedEos eos, double density_floor,
    double pressure_floor, bool eos_limiter);

void fused_cubed_sphere_sync_cuda(uint32_t** symm_signal_pads_dev,
                                  int symm_rank, int symm_world_size,
                                  torch::Device device);

void fused_cubed_sphere_flux_cuda(
    torch::Tensor w, torch::Tensor flux2, torch::Tensor flux3,
    torch::Tensor symm_buffer, void** symm_buffer_ptrs_dev,
    torch::Tensor side_meta, int face, int local_block, torch::Tensor x2v,
    torch::Tensor x2f, torch::Tensor x3v, torch::Tensor x3f,
    FusedReconScheme recon_prim, FusedReconScheme recon_vel,
    FusedRiemannSolver solver, FusedEos eos, double gammad,
    double density_floor, double pressure_floor, bool eos_limiter,
    torch::Tensor inv_mu_ratio_m1, torch::Tensor cv_ratio_m1, torch::Tensor u0,
    int shallow_roe_dir_yz);

void fused_cubed_sphere_release_cuda(uint32_t** symm_signal_pads_dev,
                                     int symm_rank, int symm_world_size,
                                     torch::Device device);

}  // namespace snap
