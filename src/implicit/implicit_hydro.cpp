// yaml
#include <yaml-cpp/yaml.h>

// snap
#include <snap/snap.h>

#include <snap/coord/coord_utils.hpp>
#include <snap/hydro/hydro.hpp>
#include <snap/mesh/meshblock.hpp>

#include "implicit_dispatch.hpp"
#include "implicit_hydro.hpp"

namespace snap {

ImplicitHydroImpl::ImplicitHydroImpl(ImplicitOptions const& options_,
                                     torch::nn::Module* p)
    : options(options_) {
  phydro = dynamic_cast<HydroImpl const*>(p);
  reset();
}

void ImplicitHydroImpl::reset() {
  TORCH_CHECK(phydro, "[ImplicitHydro] Parent Hydro is null");
}

torch::Tensor ImplicitHydroImpl::forward(torch::Tensor du, torch::Tensor w,
                                         torch::Tensor gamma, double dt) {
  if (options->scheme() == 0) {  // null operation
    return torch::zeros_like(du);
  }

  auto pcoord = phydro->pmb->pcoord;
  auto interior = phydro->pmb->part({0, 0, 0}, PartOptions().exterior(false));
  auto cos_theta = pcoord->cosine_cell_kj;
  auto sin_theta = torch::sqrt(1.0 - cos_theta * cos_theta);

  /*if (torch::isnan(du.index(interior)).any().item<bool>()) {
    TORCH_CHECK(false, "[ImplicitHydro] NaN encountered before implicit solve");
  }*/

  auto du0 = du.clone();

  /// (1) Project to local orthonormal frame
  w[IVY] += w[IVZ] * cos_theta;
  w[IVZ] *= sin_theta;

  coord_vec_raise_(du.narrow(0, IVX, 3), cos_theta);
  pcoord->prim2local1_(du);

  //// -------- Solve block-tridiagonal matrix --------- ////
  int nx1 = pcoord->options->nx1();
  int nx2 = pcoord->options->nx2();
  int nx3 = pcoord->options->nx3();

  int m = options->size();
  auto a = torch::zeros({1, nx3, nx2, nx1 * m * m}, w.options());
  auto b = torch::zeros_like(a);
  auto c = torch::zeros_like(a);
  auto delta = torch::zeros({1, nx3, nx2, nx1 * m}, w.options());

  auto iter =
      at::TensorIteratorConfig()
          .resize_outputs(false)
          .check_all_same_dtype(true)
          .declare_static_shape(du.index(interior).sizes(),
                                /*squash_dims=*/{0, 3})
          .add_owned_output(du.index(interior))
          .add_owned_input(w.index(interior))
          .add_owned_input(gamma.unsqueeze(0).index(interior))
          .add_owned_input(pcoord->face_area1().unsqueeze(0).index(interior))
          .add_owned_input(pcoord->cell_volume().unsqueeze(0).index(interior))
          .add_input(a)
          .add_input(b)
          .add_input(c)
          .add_input(delta)
          .build();

  if ((options->scheme() >> 3) & 1) {
    at::native::vic_solve_full(du.device().type(), iter, dt,
                               phydro->options->grav()->grav1(), 0);
  } else {
    at::native::vic_solve_partial(du.device().type(), iter, dt,
                                  phydro->options->grav()->grav1(), 0);
  }

  /// (3) De-project from local orthonormal frame
  w[IVZ] /= sin_theta;
  w[IVY] -= w[IVZ] * cos_theta;
  pcoord->flux2global1_(du);

  /*if (torch::isnan(du.index(interior)).any().item<bool>()) {
    TORCH_CHECK(false, "[ImplicitHydro] NaN encountered after implicit solve");
  }*/

  return du - du0;
}

std::shared_ptr<ImplicitHydroImpl> ImplicitHydroImpl::create(
    ImplicitOptions const& opts, torch::nn::Module* p,
    std::string const& name) {
  TORCH_CHECK(p != nullptr, "[ImplicitHydro] Parent module is nullptr");
  TORCH_CHECK(opts != nullptr, "[ImplicitHydro] Options pointer is nullptr");

  return p->register_module(name, ImplicitHydro(opts, p));
}

}  // namespace snap
