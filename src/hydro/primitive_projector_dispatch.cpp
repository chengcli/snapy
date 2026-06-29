// torch
#include <ATen/Dispatch.h>
#include <ATen/Parallel.h>
#include <torch/torch.h>

// snap
#include "primitive_projector.hpp"
#include "primitive_projector_dispatch.hpp"
#include "primitive_projector_impl.h"

namespace snap {

namespace {

void check_primitive_projector_args(torch::Tensor w, torch::Tensor wp,
                                    torch::Tensor psf, torch::Tensor dx1f,
                                    int is, int ie,
                                    FusedPrimitiveProjector projector,
                                    double gas_constant) {
  TORCH_CHECK(projector != FusedPrimitiveProjector::None,
              "primitive_projector_dispatch requires an active projector");
  TORCH_CHECK(w.device() == wp.device() && w.device() == psf.device() &&
                  w.device() == dx1f.device(),
              "primitive_projector_dispatch requires tensors on one device");
  TORCH_CHECK(w.is_contiguous() && wp.is_contiguous() && psf.is_contiguous() &&
                  dx1f.is_contiguous(),
              "primitive_projector_dispatch requires contiguous tensors");
  TORCH_CHECK(w.sizes() == wp.sizes(),
              "primitive_projector_dispatch requires w/wp shape match");
  TORCH_CHECK(w.dim() == 4,
              "primitive_projector_dispatch expects primitive state "
              "[nvar,nc3,nc2,nc1]");
  TORCH_CHECK(psf.dim() == 3 && psf.size(0) == w.size(1) &&
                  psf.size(1) == w.size(2) && psf.size(2) == w.size(3) + 1,
              "primitive_projector_dispatch expects psf shape "
              "[nc3,nc2,nc1+1]");
  TORCH_CHECK(dx1f.dim() == 1 && dx1f.size(0) >= w.size(3),
              "primitive_projector_dispatch expects dx1f to cover nc1 cells");
  TORCH_CHECK(is >= 0 && is < ie && ie < w.size(3) + 1,
              "primitive_projector_dispatch received invalid active x1 bounds");
  TORCH_CHECK(w.scalar_type() == wp.scalar_type() &&
                  w.scalar_type() == psf.scalar_type() &&
                  w.scalar_type() == dx1f.scalar_type(),
              "primitive_projector_dispatch requires matching tensor dtypes");
  if (projector == FusedPrimitiveProjector::Temperature) {
    TORCH_CHECK(gas_constant > 0.,
                "temperature primitive projector requires a positive gas "
                "constant");
  }
}

}  // namespace

void primitive_projector_dispatch(torch::Tensor w, torch::Tensor wp,
                                  torch::Tensor psf, torch::Tensor dx1f, int is,
                                  int ie, FusedPrimitiveProjector projector,
                                  double grav, double margin,
                                  double gas_constant) {
  check_primitive_projector_args(w, wp, psf, dx1f, is, ie, projector,
                                 gas_constant);
  at::native::call_primitive_projector(w.device().type(), w, wp, psf, dx1f, is,
                                       ie, projector, grav, margin,
                                       gas_constant);
}

void primitive_projector_cpu(torch::Tensor w, torch::Tensor wp,
                             torch::Tensor psf, torch::Tensor dx1f, int is,
                             int ie, FusedPrimitiveProjector projector,
                             double grav, double margin, double gas_constant) {
  int nvar = w.size(0);
  int nc3 = w.size(1);
  int nc2 = w.size(2);
  int nc1 = w.size(3);
  int cols = nc3 * nc2;

  AT_DISPATCH_FLOATING_TYPES(w.scalar_type(), "primitive_projector_cpu", [&] {
    auto w_data = w.data_ptr<scalar_t>();
    auto wp_data = wp.data_ptr<scalar_t>();
    auto psf_data = psf.data_ptr<scalar_t>();
    auto dx1f_data = dx1f.data_ptr<scalar_t>();

    at::parallel_for(0, cols, 0, [&](int64_t begin, int64_t end) {
      for (int64_t col = begin; col < end; ++col) {
        primitive_projector_impl(w_data, wp_data, psf_data, dx1f_data, nvar,
                                 nc3, nc2, nc1, static_cast<int>(col), is, ie,
                                 projector, scalar_t(grav), scalar_t(margin),
                                 scalar_t(gas_constant));
      }
    });
  });
}

void primitive_projector_mps(torch::Tensor w, torch::Tensor wp,
                             torch::Tensor psf, torch::Tensor dx1f, int is,
                             int ie, FusedPrimitiveProjector projector,
                             double grav, double margin, double gas_constant) {
  psf.copy_(calc_hydrostatic_pressure(w, grav, dx1f, is, ie));
  wp.copy_(w);
  wp[IPR] = calc_nonhydrostatic_pressure(w[IPR], psf, margin);
  if (projector == FusedPrimitiveProjector::Temperature) {
    wp[IDN] = w[IPR] / (w[IDN] * gas_constant);
  }
}

}  // namespace snap

namespace at::native {

DEFINE_DISPATCH(call_primitive_projector);

REGISTER_ALL_CPU_DISPATCH(call_primitive_projector,
                          &snap::primitive_projector_cpu);
REGISTER_MPS_DISPATCH(call_primitive_projector, &snap::primitive_projector_mps);

}  // namespace at::native
