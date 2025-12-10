// torch
#include <ATen/TensorIterator.h>

// snap
#include <snap/snap.h>

#include "coord_dispatch.hpp"
#include "coord_utils.hpp"

namespace snap {

void coord_vec_lower_(torch::Tensor const& vel, torch::Tensor cth) {
  torch::Tensor cosine_cell = cth;
  if (cth.dim() < vel[IV2].dim()) {
    cosine_cell = cth.unsqueeze(-1);
  }

  auto iter = at::TensorIteratorConfig()
                  .resize_outputs(false)
                  .check_all_same_dtype(true)
                  .declare_static_shape(vel[IV2].sizes())
                  .add_owned_output(vel[IV2])
                  .add_owned_output(vel[IV3])
                  .add_owned_input(cosine_cell.expand_as(vel[IV2]))
                  .build();
  at::native::call_coord_vec_lower(vel.device().type(), iter);
}

void coord_vec_raise_(torch::Tensor const& vel, torch::Tensor cth) {
  auto iter = at::TensorIteratorConfig()
                  .resize_outputs(false)
                  .check_all_same_dtype(true)
                  .declare_static_shape(vel[IV2].sizes())
                  .add_owned_output(vel[IV2])
                  .add_owned_output(vel[IV3])
                  .add_owned_input(cth)
                  .build();
  at::native::call_coord_vec_raise(vel.device().type(), iter);
}

}  // namespace snap
