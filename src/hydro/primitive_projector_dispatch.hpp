#pragma once

// torch
#include <ATen/native/DispatchStub.h>
#include <torch/torch.h>

// snap
#include <snap/snap.h>

namespace snap {

void primitive_projector_dispatch(torch::Tensor w, torch::Tensor wp,
                                  torch::Tensor psf, torch::Tensor dx1f, int is,
                                  int ie, FusedPrimitiveProjector projector,
                                  double grav, double margin,
                                  double gas_constant);

}  // namespace snap

namespace at::native {

using primitive_projector_fn = void (*)(torch::Tensor w, torch::Tensor wp,
                                        torch::Tensor psf, torch::Tensor dx1f,
                                        int is, int ie,
                                        snap::FusedPrimitiveProjector projector,
                                        double grav, double margin,
                                        double gas_constant);

DECLARE_DISPATCH(primitive_projector_fn, call_primitive_projector);

}  // namespace at::native
