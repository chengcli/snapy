// C/C++
#include <algorithm>
#include <cmath>
#include <unordered_set>

// yaml
#include <yaml-cpp/yaml.h>

// snap
#include <snap/snap.h>

#include <snap/coord/coordinate.hpp>
#include <snap/hydro/hydro.hpp>
#include <snap/layout/layout.hpp>
#include <snap/mesh/meshblock.hpp>

#include "forcing.hpp"

namespace snap {
namespace {

using torch::indexing::Slice;

struct HorizontalFluxes {
  torch::Tensor x2;
  torch::Tensor x3;
};

torch::Tensor centered_x2(torch::Tensor value, torch::Tensor width, int k0,
                          int k1, int j0, int j1, int i0, int i1) {
  auto upper = value.index(
      {Slice(), Slice(k0, k1), Slice(j0 + 1, j1 + 1), Slice(i0, i1)});
  auto lower = value.index(
      {Slice(), Slice(k0, k1), Slice(j0 - 1, j1 - 1), Slice(i0, i1)});
  auto center_width =
      width.index({Slice(k0, k1), Slice(j0, j1), Slice(i0, i1)});
  auto upper_width =
      width.index({Slice(k0, k1), Slice(j0 + 1, j1 + 1), Slice(i0, i1)});
  auto lower_width =
      width.index({Slice(k0, k1), Slice(j0 - 1, j1 - 1), Slice(i0, i1)});
  auto distance = center_width + 0.5 * (upper_width + lower_width);
  return (upper - lower) / distance.unsqueeze(0).clamp_min(1.e-30);
}

torch::Tensor centered_x3(torch::Tensor value, torch::Tensor width, int k0,
                          int k1, int j0, int j1, int i0, int i1) {
  auto upper = value.index(
      {Slice(), Slice(k0 + 1, k1 + 1), Slice(j0, j1), Slice(i0, i1)});
  auto lower = value.index(
      {Slice(), Slice(k0 - 1, k1 - 1), Slice(j0, j1), Slice(i0, i1)});
  auto center_width =
      width.index({Slice(k0, k1), Slice(j0, j1), Slice(i0, i1)});
  auto upper_width =
      width.index({Slice(k0 + 1, k1 + 1), Slice(j0, j1), Slice(i0, i1)});
  auto lower_width =
      width.index({Slice(k0 - 1, k1 - 1), Slice(j0, j1), Slice(i0, i1)});
  auto distance = center_width + 0.5 * (upper_width + lower_width);
  return (upper - lower) / distance.unsqueeze(0).clamp_min(1.e-30);
}

HorizontalFluxes horizontal_gradient_fluxes(HydroImpl const* phydro,
                                            torch::Tensor scalar,
                                            torch::Tensor density) {
  auto pmb = phydro->pmb;
  auto coord = pmb->pcoord;
  auto il = coord->il();
  auto iu = coord->iu() + 1;
  auto jl = coord->jl();
  auto ju = coord->ju() + 1;
  auto kl = coord->kl();
  auto ku = coord->ku() + 1;

  auto width2 =
      coord->center_width2().to(scalar.device(), scalar.scalar_type());
  auto width3 =
      coord->center_width3().to(scalar.device(), scalar.scalar_type());
  auto flux2 = torch::zeros_like(scalar);
  auto flux3 = torch::zeros_like(scalar);

  // x2 faces: j=[jl, ju+1], with the transverse x3 derivative averaged
  // from the cells on both sides of the face.
  auto d2 =
      (scalar.index(
           {Slice(), Slice(kl, ku), Slice(jl, ju + 1), Slice(il, iu)}) -
       scalar.index(
           {Slice(), Slice(kl, ku), Slice(jl - 1, ju), Slice(il, iu)})) /
      (0.5 * (width2.index({Slice(kl, ku), Slice(jl, ju + 1), Slice(il, iu)}) +
              width2.index({Slice(kl, ku), Slice(jl - 1, ju), Slice(il, iu)})))
          .unsqueeze(0)
          .clamp_min(1.e-30);
  auto d3 = 0.5 * (centered_x3(scalar, width3, kl, ku, jl, ju + 1, il, iu) +
                   centered_x3(scalar, width3, kl, ku, jl - 1, ju, il, iu));
  auto c2 =
      coord->cosine_face2_kj.index({Slice(kl, ku), Slice(jl, ju + 1), Slice()})
          .to(scalar.device(), scalar.scalar_type());
  auto s2 = torch::sqrt((1. - c2.square()).clamp_min(1.e-30));
  auto grad2 = (d2 - c2.unsqueeze(0) * d3) / s2.unsqueeze(0);
  auto rho2 =
      0.5 * (density.index({Slice(kl, ku), Slice(jl, ju + 1), Slice(il, iu)}) +
             density.index({Slice(kl, ku), Slice(jl - 1, ju), Slice(il, iu)}));
  flux2.index_put_({Slice(), Slice(kl, ku), Slice(jl, ju + 1), Slice(il, iu)},
                   rho2.unsqueeze(0) * grad2);

  // x3 faces: k=[kl, ku+1], with the transverse x2 derivative averaged
  // from the cells on both sides of the face.
  auto d3n =
      (scalar.index(
           {Slice(), Slice(kl, ku + 1), Slice(jl, ju), Slice(il, iu)}) -
       scalar.index(
           {Slice(), Slice(kl - 1, ku), Slice(jl, ju), Slice(il, iu)})) /
      (0.5 * (width3.index({Slice(kl, ku + 1), Slice(jl, ju), Slice(il, iu)}) +
              width3.index({Slice(kl - 1, ku), Slice(jl, ju), Slice(il, iu)})))
          .unsqueeze(0)
          .clamp_min(1.e-30);
  auto d2t = 0.5 * (centered_x2(scalar, width2, kl, ku + 1, jl, ju, il, iu) +
                    centered_x2(scalar, width2, kl - 1, ku, jl, ju, il, iu));
  auto c3 =
      coord->cosine_face3_kj.index({Slice(kl, ku + 1), Slice(jl, ju), Slice()})
          .to(scalar.device(), scalar.scalar_type());
  auto s3 = torch::sqrt((1. - c3.square()).clamp_min(1.e-30));
  auto grad3 = (d3n - c3.unsqueeze(0) * d2t) / s3.unsqueeze(0);
  auto rho3 =
      0.5 * (density.index({Slice(kl, ku + 1), Slice(jl, ju), Slice(il, iu)}) +
             density.index({Slice(kl - 1, ku), Slice(jl, ju), Slice(il, iu)}));
  flux3.index_put_({Slice(), Slice(kl, ku + 1), Slice(jl, ju), Slice(il, iu)},
                   rho3.unsqueeze(0) * grad3);

  return {flux2, flux3};
}

}  // namespace

ScalarHyperdiffusionOptions ScalarHyperdiffusionOptionsImpl::from_yaml(
    YAML::Node const& forcing) {
  if (!forcing["scalar-hyperdiffusion"]) return nullptr;

  auto node = forcing["scalar-hyperdiffusion"];
  auto op = ScalarHyperdiffusionOptionsImpl::create();
  op->damping_time() = node["damping-time"].as<double>(0.);
  op->fields() =
      node["fields"].as<std::vector<std::string>>(std::vector<std::string>{});

  TORCH_CHECK(op->damping_time() > 0.,
              "ScalarHyperdiffusionOptions: damping-time must be positive.");
  TORCH_CHECK(!op->fields().empty(),
              "ScalarHyperdiffusionOptions: fields must not be empty.");
  std::unordered_set<std::string> unique;
  for (auto const& field : op->fields()) {
    TORCH_CHECK(unique.insert(field).second,
                "ScalarHyperdiffusionOptions: duplicate field '", field, "'.");
  }
  return op;
}

torch::Tensor ScalarLaplacianImpl::forward(torch::Tensor scalar,
                                           torch::Tensor density) const {
  TORCH_CHECK(phydro, "[ScalarLaplacian] Parent Hydro is null");
  TORCH_CHECK(scalar.dim() == 4,
              "[ScalarLaplacian] scalar must have shape [n,k,j,i]");
  TORCH_CHECK(density.dim() == 3,
              "[ScalarLaplacian] density must have shape [k,j,i]");
  auto flux = horizontal_gradient_fluxes(phydro, scalar, density);
  return phydro->pmb->pcoord->divergence(torch::Tensor(), flux.x2, flux.x3) /
         density.unsqueeze(0).clamp_min(1.e-30);
}

ScalarHyperdiffusionImpl::ScalarHyperdiffusionImpl(
    ScalarHyperdiffusionOptions const& options_, torch::nn::Module* p)
    : options(options_) {
  phydro = dynamic_cast<HydroImpl const*>(p);
  laplacian = register_module("laplacian", ScalarLaplacian(phydro));
  reset();
}

void ScalarHyperdiffusionImpl::reset() {
  TORCH_CHECK(phydro, "[ScalarHyperdiffusion] Parent Hydro is null");
  auto pmb = phydro->pmb;
  auto coord = pmb->pcoord;
  TORCH_CHECK(coord->options->type() == "gnomonic-equiangle" &&
                  pmb->get_layout()->options->type() == "cubed-sphere",
              "[ScalarHyperdiffusion] gnomonic-equiangle coordinates with a "
              "cubed-sphere layout are required.");
  TORCH_CHECK(coord->options->nghost() >= 1,
              "[ScalarHyperdiffusion] nghost must be at least 1.");
  TORCH_CHECK(
      coord->options->nx2() > 1 && coord->options->nx3() > 1,
      "[ScalarHyperdiffusion] both horizontal dimensions are required.");

  auto thermo = phydro->options->eos()->thermo();
  hydro_ids.clear();
  vel1_field = -1;
  std::vector<std::string> species =
      thermo ? thermo->species() : std::vector<std::string>{};
  for (int n = 0; n < options->fields().size(); ++n) {
    auto const& field = options->fields()[n];
    if (field == "vel1") {
      vel1_field = n;
      hydro_ids.push_back(IVX);
      continue;
    }
    auto it = std::find(species.begin(), species.end(), field);
    TORCH_CHECK(it != species.end(),
                "[ScalarHyperdiffusion] Unknown vapor or "
                "cloud species '",
                field, "'.");
    auto id = std::distance(species.begin(), it);
    TORCH_CHECK(id > 0, "[ScalarHyperdiffusion] Dry species '", field,
                "' cannot be diffused.");
    hydro_ids.push_back(ICY + id - 1);
  }

  // Build the reference bound from the complete panel, not the local block,
  // so decomposition does not change K4 at block boundaries.
  auto op = coord->options;
  auto alpha_f = torch::linspace(op->global_x2min(), op->global_x2max(),
                                 op->global_nx2() + 1, torch::kFloat64);
  auto beta_f = torch::linspace(op->global_x3min(), op->global_x3max(),
                                op->global_nx3() + 1, torch::kFloat64);
  auto alpha = 0.5 * (alpha_f.slice(0, 0, op->global_nx2()) +
                      alpha_f.slice(0, 1, op->global_nx2() + 1));
  auto beta = 0.5 * (beta_f.slice(0, 0, op->global_nx3()) +
                     beta_f.slice(0, 1, op->global_nx3() + 1));
  auto x = alpha.tan().unsqueeze(0);
  auto y = beta.tan().unsqueeze(1);
  auto xf0 = alpha_f.slice(0, 0, op->global_nx2()).tan().unsqueeze(0);
  auto xf1 = alpha_f.slice(0, 1, op->global_nx2() + 1).tan().unsqueeze(0);
  auto yf0 = beta_f.slice(0, 0, op->global_nx3()).tan().unsqueeze(1);
  auto yf1 = beta_f.slice(0, 1, op->global_nx3() + 1).tan().unsqueeze(1);
  auto delta = torch::sqrt(1. + x.square() + y.square());
  auto sine =
      delta / (torch::sqrt(1. + x.square()) * torch::sqrt(1. + y.square()));
  auto dx2_angle = torch::acos(((1. + xf0 * xf1 + y.square()) /
                                (torch::sqrt(1. + xf0.square() + y.square()) *
                                 torch::sqrt(1. + xf1.square() + y.square())))
                                   .clamp(-1., 1.));
  auto dx3_angle = torch::acos(((1. + x.square() + yf0 * yf1) /
                                (torch::sqrt(1. + x.square() + yf0.square()) *
                                 torch::sqrt(1. + x.square() + yf1.square())))
                                   .clamp(-1., 1.));
  auto h2_unit_min = (dx2_angle * sine).min();
  auto h3_unit_min = (dx3_angle * sine).min();
  auto radius = coord->x1v;
  auto inv_h2 = (radius * h2_unit_min).square().reciprocal();
  auto inv_h3 = (radius * h3_unit_min).square().reciprocal();
  auto lambda_interior = 4. * (inv_h2 + inv_h3);

  lambda_grid = register_buffer("lambda_grid", lambda_interior);
  k4 = register_buffer("k4",
                       1. / (options->damping_time() * lambda_grid.square()));
}

torch::Tensor ScalarHyperdiffusionImpl::forward(torch::Tensor du,
                                                torch::Tensor w,
                                                torch::Tensor temp, double dt) {
  (void)temp;
  auto pmb = phydro->pmb;
  auto coord = pmb->pcoord;
  auto density = w[IDN];

  std::vector<torch::Tensor> selected;
  selected.reserve(hydro_ids.size());
  for (auto id : hydro_ids) selected.push_back(w[id]);
  auto scalar = torch::stack(selected);
  auto intermediate = laplacian->forward(scalar, density);

  Variables exchange_vars;
  exchange_vars["scalar_hyperdiffusion_laplacian"] = intermediate;
  SyncOptions sync;
  sync.interpolate(true).type(kScalar).phyid(19);
  pmb->exchange(exchange_vars, sync);

  auto flux = horizontal_gradient_fluxes(phydro, intermediate, density);
  auto coeff = k4.to(w.device(), w.scalar_type()).view({1, 1, 1, -1});
  flux.x2.mul_(coeff);
  flux.x3.mul_(coeff);
  auto tendency = -coord->divergence(torch::Tensor(), flux.x2, flux.x3);
  for (int n = 0; n < hydro_ids.size(); ++n) {
    du[hydro_ids[n]].add_(dt * tendency[n]);
  }

  if (vel1_field >= 0) {
    auto il = coord->il();
    auto iu = coord->iu() + 1;
    auto jl = coord->jl();
    auto ju = coord->ju() + 1;
    auto kl = coord->kl();
    auto ku = coord->ku() + 1;
    auto eflux2 = torch::zeros_like(w[IDN]).unsqueeze(0);
    auto eflux3 = torch::zeros_like(w[IDN]).unsqueeze(0);
    auto v2 =
        0.5 * (w[IVX].index({Slice(kl, ku), Slice(jl, ju + 1), Slice(il, iu)}) +
               w[IVX].index({Slice(kl, ku), Slice(jl - 1, ju), Slice(il, iu)}));
    auto v3 =
        0.5 * (w[IVX].index({Slice(kl, ku + 1), Slice(jl, ju), Slice(il, iu)}) +
               w[IVX].index({Slice(kl - 1, ku), Slice(jl, ju), Slice(il, iu)}));
    eflux2.index_put_(
        {0, Slice(kl, ku), Slice(jl, ju + 1), Slice(il, iu)},
        v2 * flux.x2[vel1_field].index(
                 {Slice(kl, ku), Slice(jl, ju + 1), Slice(il, iu)}));
    eflux3.index_put_(
        {0, Slice(kl, ku + 1), Slice(jl, ju), Slice(il, iu)},
        v3 * flux.x3[vel1_field].index(
                 {Slice(kl, ku + 1), Slice(jl, ju), Slice(il, iu)}));
    du[IPR].sub_(dt * coord->divergence(torch::Tensor(), eflux2, eflux3)[0]);
  }
  return du;
}

double ScalarHyperdiffusionImpl::max_time_step(torch::Tensor w) const {
  auto coord = phydro->pmb->pcoord;
  auto rho = w[IDN];
  auto center = rho.index({Slice(coord->kl(), coord->ku() + 1),
                           Slice(coord->jl(), coord->ju() + 1),
                           Slice(coord->il(), coord->iu() + 1)})
                    .clamp_min(1.e-30);
  auto max_face_ratio = torch::ones_like(center);
  for (int offset : {-1, 1}) {
    auto neighbor2 =
        rho.index({Slice(coord->kl(), coord->ku() + 1),
                   Slice(coord->jl() + offset, coord->ju() + 1 + offset),
                   Slice(coord->il(), coord->iu() + 1)});
    auto neighbor3 =
        rho.index({Slice(coord->kl() + offset, coord->ku() + 1 + offset),
                   Slice(coord->jl(), coord->ju() + 1),
                   Slice(coord->il(), coord->iu() + 1)});
    max_face_ratio = torch::maximum(
        max_face_ratio, torch::maximum(0.5 * (center + neighbor2) / center,
                                       0.5 * (center + neighbor3) / center));
  }
  auto ratio = max_face_ratio.max().item<double>();
  return 2. * options->damping_time() / (ratio * ratio);
}

}  // namespace snap
