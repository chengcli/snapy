// C/C++
#include <algorithm>
#include <array>
#include <limits>

// yaml
#include <yaml-cpp/yaml.h>

// snap
#include <snap/snap.h>

#include <snap/coord/coordinate.hpp>
#include <snap/hydro/hydro.hpp>
#include <snap/mesh/meshblock.hpp>

#include "forcing.hpp"

namespace snap {
namespace {

constexpr int kSpatialDims[] = {3, 2, 1};
using Region = std::array<int64_t, 3>;

torch::Tensor cell_width(Coordinate const& coord, int idir) {
  if (idir == 0) return coord->center_width1();
  if (idir == 1) return coord->center_width2();
  return coord->center_width3();
}

torch::Tensor center_distance(Coordinate const& coord, int idir) {
  if (idir == 0) return coord->center_distance1();
  if (idir == 1) return coord->center_distance2();
  return coord->center_distance3();
}

std::vector<torch::indexing::TensorIndex> region_index(Region const& start,
                                                       Region const& end) {
  return {torch::indexing::Slice(start[0], end[0]),
          torch::indexing::Slice(start[1], end[1]),
          torch::indexing::Slice(start[2], end[2])};
}

torch::Tensor region_slice(torch::Tensor value, Region const& start,
                           Region const& end) {
  return value.index(region_index(start, end));
}

std::pair<Region, Region> region_bounds(
    std::vector<torch::indexing::TensorIndex> const& index) {
  Region start;
  Region end;
  for (int dim = 0; dim < 3; ++dim) {
    start[dim] = index[dim].slice().start().expect_int();
    end[dim] = index[dim].slice().stop().expect_int();
  }
  return {start, end};
}

torch::Tensor spacing_slice(torch::Tensor spacing, int idir,
                            Region const& start, Region const& end) {
  auto dim = kSpatialDims[idir] - 1;
  return spacing.slice(dim, start[dim], end[dim]);
}

torch::Tensor centered_derivative(torch::Tensor value, Coordinate const& coord,
                                  int idir, Region start, Region end) {
  auto dim = kSpatialDims[idir] - 1;
  auto upper_start = start;
  auto upper = end;
  auto lower = start;
  auto lower_end = end;
  ++upper_start[dim];
  ++upper[dim];
  --lower[dim];
  --lower_end[dim];
  return (region_slice(value, upper_start, upper) -
          region_slice(value, lower, lower_end)) /
         (2. * spacing_slice(cell_width(coord, idir), idir, start, end));
}

torch::Tensor face_normal_derivative(torch::Tensor value,
                                     Coordinate const& coord, int idir,
                                     Region start, Region end) {
  auto dim = kSpatialDims[idir] - 1;
  auto lower = start;
  auto lower_end = end;
  --lower[dim];
  --lower_end[dim];
  return (region_slice(value, start, end) -
          region_slice(value, lower, lower_end)) /
         spacing_slice(center_distance(coord, idir), idir, start, end);
}

torch::Tensor face_average(torch::Tensor value, int idir, Region start,
                           Region end) {
  auto dim = kSpatialDims[idir] - 1;
  auto lower = start;
  auto lower_end = end;
  --lower[dim];
  --lower_end[dim];
  return 0.5 * (region_slice(value, start, end) +
                region_slice(value, lower, lower_end));
}

//! Linear extrapolation of `value` onto a physical wall face from the two
//! nearest ACTIVE cells. Weights come from the actual cell widths, so a
//! stretched mesh is handled correctly. Falls back to the nearest active cell
//! if the extrapolation is non-positive; that value also reads no ghost.
torch::Tensor extrapolate_to_wall(torch::Tensor value, Coordinate const& coord,
                                  int idir, Region start, Region end,
                                  bool upper) {
  auto dim = kSpatialDims[idir] - 1;
  auto near = start;
  auto near_end = end;
  auto next = start;
  auto next_end = end;
  near[dim] = upper ? end[dim] - 2 : start[dim];
  next[dim] = upper ? end[dim] - 3 : start[dim] + 1;
  near_end[dim] = near[dim] + 1;
  next_end[dim] = next[dim] + 1;

  auto width = cell_width(coord, idir);
  auto wnear = spacing_slice(width, idir, near, near_end);
  auto wnext = spacing_slice(width, idir, next, next_end);
  auto t = wnear / (wnear + wnext);

  auto vnear = region_slice(value, near, near_end);
  auto vnext = region_slice(value, next, next_end);
  auto wall = (1. + t) * vnear - t * vnext;
  return torch::where(wall > 0., wall, vnear);
}

//! Face-centred diffusion coefficient. Interior faces take the two-cell
//! average; a physical wall face is extrapolated from active cells instead,
//! because the ghost there is filled by a boundary condition to serve a
//! different operator and is not the physical state at the wall.
torch::Tensor face_coefficient(torch::Tensor value, Coordinate const& coord,
                               int idir, Region start, Region end,
                               bool wall_lower, bool wall_upper) {
  auto out = face_average(value, idir, start, end);
  auto dim = kSpatialDims[idir] - 1;
  auto nface = end[dim] - start[dim];
  if (nface < 3) return out;  // fewer than two active cells to extrapolate from

  if (wall_lower) {
    out.narrow(dim, 0, 1).copy_(
        extrapolate_to_wall(value, coord, idir, start, end, false));
  }
  if (wall_upper) {
    out.narrow(dim, nface - 1, 1)
        .copy_(extrapolate_to_wall(value, coord, idir, start, end, true));
  }
  return out;
}

bool active(Coordinate const& coord, int idir) {
  if (idir == 0) return coord->options->nc1() > 1;
  if (idir == 1) return coord->options->nc2() > 1;
  return coord->options->nc3() > 1;
}

}  // namespace

DiffusionOptions DiffusionOptionsImpl::from_yaml(YAML::Node const& forcing) {
  if (!forcing["diffusion"]) return nullptr;

  auto node = forcing["diffusion"];
  TORCH_CHECK(!node["K"] && !node["type"],
              "DiffusionOptions: legacy 'K' and 'type' keys are unsupported; "
              "use 'nu_iso' and 'kappa_iso'.");

  auto op = DiffusionOptionsImpl::create();
  op->nu_iso() = node["nu_iso"].as<double>(0.);
  op->kappa_iso() = node["kappa_iso"].as<double>(0.);
  TORCH_CHECK(op->nu_iso() >= 0.,
              "DiffusionOptions: nu_iso must be non-negative.");
  TORCH_CHECK(op->kappa_iso() >= 0.,
              "DiffusionOptions: kappa_iso must be non-negative.");
  return op;
}

DiffusionImpl::DiffusionImpl(DiffusionOptions const& options_,
                             torch::nn::Module* p)
    : options(options_) {
  phydro = dynamic_cast<HydroImpl const*>(p);
  reset();
}

void DiffusionImpl::reset() {
  TORCH_CHECK(phydro, "[Diffusion] Parent Hydro is null");

  auto coord = phydro->pmb->pcoord;
  auto enabled = options->nu_iso() > 0. || options->kappa_iso() > 0.;
  TORCH_CHECK(!enabled || coord->options->type() == "cartesian",
              "[Diffusion] Only cartesian coordinates are supported.");
  TORCH_CHECK(options->nu_iso() == 0. || coord->options->nghost() >= 2,
              "[Diffusion] Isotropic viscosity requires nghost >= 2.");
  TORCH_CHECK(options->kappa_iso() == 0. || phydro->peos->species_cv_ref() > 0.,
              "[Diffusion] Isotropic heat conduction requires an EOS with a "
              "positive reference specific heat at constant volume.");
}

torch::Tensor DiffusionImpl::forward(torch::Tensor du, torch::Tensor w,
                                     torch::Tensor temp, double dt) {
  auto pmb = phydro->pmb;
  auto coord = pmb->pcoord;
  auto [cell_start, cell_end] = region_bounds(
      pmb->part({0, 0, 0}, PartOptions().exterior(false).ndim(3)));
  PartOptions div_options;
  div_options.exterior(false).ndim(3);
  for (int idir = 0; idir < 3; ++idir) {
    if (!active(coord, idir)) continue;
    if (idir == 0) div_options.extend_x1(1);
    if (idir == 1) div_options.extend_x2(1);
    if (idir == 2) div_options.extend_x3(1);
  }
  auto [div_start, div_end] = region_bounds(pmb->part({0, 0, 0}, div_options));

  torch::Tensor div_vel;
  torch::Tensor rho_cv;

  if (options->nu_iso() > 0.) {
    div_vel = torch::zeros_like(w[IDN]);
    auto div_vel_region = region_slice(div_vel, div_start, div_end);
    for (int idir = 0; idir < 3; ++idir) {
      if (active(coord, idir)) {
        div_vel_region +=
            centered_derivative(w[IVX + idir], coord, idir, div_start, div_end);
      }
    }
  }
  if (options->kappa_iso() > 0.) {
    rho_cv = w[IDN] * phydro->peos->specific_heat_cv(w, temp);
  }

  std::array<torch::Tensor, 3> fluxes;
  bool has_flux = false;
  for (int idir = 0; idir < 3; ++idir) {
    if (!active(coord, idir)) continue;
    has_flux = true;

    auto dim = kSpatialDims[idir] - 1;
    auto face_start = cell_start;
    auto face_end = cell_end;
    ++face_end[dim];
    auto face_index = region_index(face_start, face_end);
    auto flux = torch::zeros_like(w);

    // x1 only: the vertical is where a reflecting wall cuts a monotone,
    // gravity-stratified profile. A lateral reflecting face may be a true
    // symmetry plane, where the two-cell average is the better estimate, and
    // reflecting is the default for every unspecified face, so extending this
    // to x2/x3 would opt configurations in silently.
    bool wall_lower = idir == 0 && pmb->options->is_wall_boundary(0, 0, -1);
    bool wall_upper = idir == 0 && pmb->options->is_wall_boundary(0, 0, 1);

    auto rho_face = face_coefficient(w[IDN], coord, idir, face_start, face_end,
                                     wall_lower, wall_upper);

    if (options->nu_iso() > 0.) {
      auto div_face = face_average(div_vel, idir, face_start, face_end);
      for (int ivar = 0; ivar < 3; ++ivar) {
        auto stress = face_normal_derivative(w[IVX + ivar], coord, idir,
                                             face_start, face_end);
        if (ivar == idir) {
          stress = 2. * stress - (2. / 3.) * div_face;
        } else if (active(coord, ivar)) {
          auto shear_start = face_start;
          auto shear_end = face_end;
          --shear_start[dim];
          --shear_end[dim];
          stress += 0.5 * (centered_derivative(w[IVX + idir], coord, ivar,
                                               face_start, face_end) +
                           centered_derivative(w[IVX + idir], coord, ivar,
                                               shear_start, shear_end));
        }

        auto momentum_flux = -options->nu_iso() * rho_face * stress;
        flux[IVX + ivar].index(face_index).copy_(momentum_flux);
        flux[IPR].index(face_index) +=
            face_average(w[IVX + ivar], idir, face_start, face_end) *
            momentum_flux;
      }
    }

    if (options->kappa_iso() > 0.) {
      // W->T is in kelvin, so convert thermal diffusivity to conductivity
      // using the local mixture volumetric heat capacity.
      flux[IPR].index(face_index) -=
          options->kappa_iso() *
          face_coefficient(rho_cv, coord, idir, face_start, face_end,
                           wall_lower, wall_upper) *
          face_normal_derivative(temp, coord, idir, face_start, face_end);
    }
    fluxes[idir] = flux;
  }

  if (has_flux) {
    du -= dt * coord->divergence(fluxes[0], fluxes[1], fluxes[2]);
  }
  return du;
}

double DiffusionImpl::max_time_step(torch::Tensor w) const {
  auto coord = phydro->pmb->pcoord;
  auto interior =
      phydro->pmb->part({0, 0, 0}, PartOptions().exterior(false).ndim(3));
  int ndim = 0;
  double dx_min = std::numeric_limits<double>::max();

  for (int idir = 0; idir < 3; ++idir) {
    if (!active(coord, idir)) continue;
    ++ndim;
    dx_min = std::min(dx_min, cell_width(coord, idir)
                                  .expand_as(w[IDN])
                                  .index(interior)
                                  .min()
                                  .item<double>());
  }

  if (ndim == 0) return std::numeric_limits<double>::max();
  auto coeff = std::max(options->nu_iso(), options->kappa_iso());
  if (coeff == 0.) return std::numeric_limits<double>::max();
  return dx_min * dx_min / (2. * ndim * coeff);
}

}  // namespace snap
