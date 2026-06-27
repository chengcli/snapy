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
    torch::Tensor rho_grav);

void fused_cubed_sphere_exchange_cuda(
    torch::Tensor w, torch::Tensor flux2, torch::Tensor flux3,
    torch::Tensor symm_buffer, void** symm_buffer_ptrs_dev,
    uint32_t** symm_signal_pads_dev, int face, int symm_rank,
    int symm_world_size, torch::Tensor side_meta, torch::Tensor x2v,
    torch::Tensor x2f, torch::Tensor x3v, torch::Tensor x3f,
    FusedReconScheme recon_prim, FusedReconScheme recon_vel,
    FusedRiemannSolver solver, FusedEos eos, double gammad,
    double density_floor, double pressure_floor, bool eos_limiter,
    torch::Tensor inv_mu_ratio_m1, torch::Tensor cv_ratio_m1, torch::Tensor u0,
    int shallow_roe_dir_yz, FusedPrimitiveProjector projector);

}  // namespace snap
