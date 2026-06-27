#pragma once

// torch
#include <torch/torch.h>

namespace snap {

enum class FusedReconScheme : int {
  CP3 = 0,
  CP5 = 1,
  WENO3 = 2,
  WENO5 = 3,
};

enum class FusedRiemannSolver : int {
  LMARS = 0,
  HLLC = 1,
};

enum class FusedEos : int {
  IdealGas = 0,
  IdealMoist = 1,
};

void fused_recon_riemann_cuda(torch::Tensor w, torch::Tensor flux, int dim,
                              FusedReconScheme recon_prim,
                              FusedReconScheme recon_vel,
                              FusedRiemannSolver solver, FusedEos eos,
                              double gammad, double density_floor,
                              double pressure_floor, bool eos_limiter,
                              torch::Tensor inv_mu_ratio_m1,
                              torch::Tensor cv_ratio_m1, torch::Tensor u0);

}  // namespace snap
