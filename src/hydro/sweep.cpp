#include "sweep.hpp"

// C/C++
#include <algorithm>

// snap
#include <snap/eb/geometry.hpp>
#include <snap/eb/sphere_reconstruction.hpp>
#include <snap/eb/sphere_transport.hpp>
#include <snap/eos/ideal_moist.hpp>
#include <snap/mesh/meshblock.hpp>

#include "hydro.hpp"

namespace snap {

// The position and least-squares coefficients are fixed by the fluid
// polyhedra. Store only fluid cells whose reconstructed states can replace a
// regular face stencil. For a closed face the neighbour index is the cell
// itself, which also excludes that neighbour from the local extrema.
struct CutCellStencil {
  torch::Tensor cells, positions;
  std::vector<torch::Tensor> neighbours, coefficients;
  int width;
  std::shared_ptr<SphereStencil> sphere;
  explicit CutCellStencil(int w) : width(w) {}

  CutCellStencil(EBGeometry const& g, torch::Tensor const& state, int w)
      : width(w) {
    auto irregular = g.flags.ne(0);
    auto affected = irregular.clone();
    for (int axis = 0; axis < 3; ++axis) {
      if (irregular.size(axis) == 1) continue;
      for (int shift = -w - 1; shift <= w + 1; ++shift)
        affected.logical_or_(irregular.roll(shift, axis));
    }
    cells = (affected & g.volume_fraction.gt(0.)).flatten().nonzero().flatten();
    auto x = g.cell_centroid.flatten(1);
    positions = x.index_select(1, cells);
    auto ids = torch::arange(irregular.numel(), cells.options())
                   .reshape(irregular.sizes());
    auto matrix = torch::zeros({cells.numel(), 3, 3}, x.options());
    for (int d = 0; d < 3; ++d) {
      int axis = 2 - d, n = irregular.size(axis);
      if (n == 1) continue;
      for (int step : {-1, 1}) {
        auto open = g.face_fraction[d].narrow(axis, step > 0, n) > 0.;
        open.select(axis, step > 0 ? n - 1 : 0).fill_(false);
        auto neighbour = torch::where(open, ids.roll(-step, axis), ids)
                             .flatten()
                             .index_select(0, cells);
        auto dx = (x.index_select(1, neighbour) - positions).t();
        auto dist2 = dx.square().sum(1, true);
        auto weighted = dx / torch::where(dist2 > 0., dist2, 1.);
        matrix.add_(weighted.unsqueeze(-1) * dx.unsqueeze(-2));
        coefficients.push_back(weighted.unsqueeze(-1));
        neighbours.push_back(neighbour);
      }
    }
    // A missing direction (including a degenerate axis) has zero slope.
    auto inverse = torch::linalg_pinv(matrix, 1.e-12, true);
    for (auto& c : coefficients)
      c = torch::matmul(inverse, c).squeeze(-1).t().to(state.options());
  }
};

CutCellReconstruction::CutCellReconstruction(EBGeometry const& geometry,
                                             torch::Tensor const& state,
                                             int width,
                                             MeshBlockImpl const* sphere,
                                             bool velocity)
    : _geometry(geometry), _state(state) {
  if (sphere) {
    auto& cached = geometry.reconstruction_stencil;
    if (!cached || !cached->sphere || cached->width < width) {
      cached = std::make_shared<CutCellStencil>(width);
      cached->sphere =
          std::make_shared<SphereStencil>(*sphere, geometry, width);
    }
    _stencil = cached;
    _mapped = std::make_shared<SphereReconstruction>(*cached->sphere, state,
                                                     velocity);
    return;
  }
  auto& cached = geometry.reconstruction_stencil;
  // EBGeometry::to and set_embedded_boundary install fresh geometry objects.
  if (!cached || cached->width < width ||
      cached->cells.device() != state.device() ||
      (!cached->coefficients.empty() &&
       cached->coefficients.front().scalar_type() != state.scalar_type()))
    cached = std::make_shared<CutCellStencil>(geometry, state, width);
  _stencil = cached;
  auto flat = _state.flatten(1);
  _packed = flat.index_select(1, cached->cells);
  _lower = _upper = _packed;
  _gradient =
      torch::zeros({state.size(0), 3, cached->cells.numel()}, state.options());
  for (size_t i = 0; i < cached->neighbours.size(); ++i) {
    auto neighbour = flat.index_select(1, cached->neighbours[i]);
    _lower = torch::minimum(_lower, neighbour);
    _upper = torch::maximum(_upper, neighbour);
    _gradient.add_((neighbour - _packed).unsqueeze(1) *
                   cached->coefficients[i]);
  }
}

torch::Tensor CutCellReconstruction::evaluate(
    torch::Tensor const& points) const {
  auto const& ids = _stencil->cells;
  auto dx = (points.flatten(1).index_select(1, ids) - _stencil->positions)
                .to(_state.options());
  auto value = torch::clamp(_packed + (_gradient * dx).sum(1), _lower, _upper);
  auto out = _state.contiguous().clone();
  out.flatten(1).index_copy_(1, ids, value);
  return out;
}

torch::Tensor CutCellReconstruction::faces(torch::Tensor const& regular,
                                           int dim, int width) const {
  if (_mapped) return _mapped->faces(regular, dim);
  int d = 3 - dim, n = _state.size(dim);
  auto const& centroid = _geometry.face_centroid[d];
  auto left = evaluate(centroid.narrow(dim, 1, n)).roll(1, dim);
  auto right = evaluate(centroid.narrow(dim, 0, n));
  return torch::where(irregular_faces(dim, width), torch::stack({left, right}),
                      regular);
}

torch::Tensor CutCellReconstruction::irregular_faces(int dim, int width) const {
  int d = 3 - dim, n = _state.size(dim);
  auto irregular = _geometry.flags.ne(0);
  auto use = _geometry.face_fraction[d].narrow(dim - 1, 0, n).ne(1.);
  // The stencil for both sides spans the face, donor cells i-1 and i.
  for (int offset = 1 - width; offset <= width; ++offset)
    use.logical_or_(irregular.roll(offset, dim - 1));
  return use;
}

torch::Tensor CutCellReconstruction::wall_state() const {
  return _mapped ? _mapped->wall() : evaluate(_geometry.wall_centroid);
}

void post_face_floors(ReconstructImpl const& recon, RegisterMonitor& monitor) {
  if (!recon.floored.defined()) return;
  monitor.count("face_floor.rho", recon.floored[0]);
  monitor.count("face_floor.p", recon.floored[1]);
  monitor.count("face_floor.species", recon.floored[2]);
}

torch::Tensor reconstruct_faces(HydroImpl& hydro, FaceDir const& dir,
                                torch::Tensor const& w,
                                IdealMoistPerturbations const* perturb,
                                torch::Tensor& background) {
  enum { DIM1 = 3, DIM2 = 2, DIM3 = 1 };
  int dim = dir.dim;
  auto& monitor = hydro.flux_register().monitor();
  auto recon = dim == DIM1 ? hydro.precon1 : hydro.precon23;
  if (!hydro.fp32()) {
    auto lr = recon->forward(w, dim);
    post_face_floors(*recon, monitor);
    return lr;
  }
  // The gauge: the midrange of pressure0 along the sweep. The mean is a
  // rounded sum over the block's own cells, so on a column that is uniform
  // along the sweep it is 1 ulp off by block extent and the FP32 face states
  // depend on the decomposition. The midrange of a uniform column is the
  // column's value exactly in every block, and elsewhere it bounds
  // |pressure0 - gauge| at half the range, as small as any constant can.
  auto [lo, hi] = torch::aminmax(perturb->pressure0, dim - 1, true);
  auto gauge = 0.5 * (lo + hi);
  auto input = w.to(torch::kFloat32);
  input[IPR].copy_((perturb->pressure0 - gauge) + perturb->pressure_delta);
  auto lr = recon->forward(input, dim, false);
  post_face_floors(*recon, monitor);
  // FaceFloor on the FP32 split path: the total face states this path
  // floors itself, on the faces the reconstruction computes
  auto count_below = [&](torch::Tensor const& x, double floor, int axis) {
    int ng = recon->pinterp1->stencils() / 2 + 1;
    int64_t n = x.size(axis);
    int64_t il = ng, iu = n - ng;
    return (x < floor).narrow(axis, il, std::min<int64_t>(iu + 2, n) - il);
  };
  // Horizontal Cartesian sweeps need FP64 only for pressure cancellation.
  // Covariance consumers still require total face states; retain their path.
  bool active_horizontal = (dim == DIM2 && !hydro.options->disable_flux_x2()) ||
                           (dim == DIM3 && !hydro.options->disable_flux_x3());
  if (active_horizontal && !hydro.flux_covariance()) {
    int axis = dim - 1;
    auto p0 = perturb->pressure0;
    int n = p0.size(axis);
    auto below =
        torch::cat({p0.narrow(axis, 0, 1), p0.narrow(axis, 0, n - 1)}, axis);
    background = 0.5 * (p0 + below);
    auto pressure = lr.select(1, IPR).to(torch::kFloat64);
    pressure += gauge;
    monitor.count(
        "face_floor.p",
        count_below(pressure, hydro.peos->options->pressure_floor(), dim));
    pressure.clamp_min_(hydro.peos->options->pressure_floor());
    pressure -= background;
    lr.select(1, IPR).copy_(pressure);
    monitor.count("face_floor.rho",
                  count_below(lr.select(1, IDN),
                              hydro.peos->options->density_floor(), dim));
    lr.select(1, IDN).clamp_min_(hydro.peos->options->density_floor());
    return lr;
  }
  lr = lr.to(torch::kFloat64);
  for (int side : {ILT, IRT}) {
    lr[side][IPR] += gauge;
    monitor.count("face_floor.p",
                  count_below(lr[side][IPR],
                              hydro.peos->options->pressure_floor(), dim - 1));
    monitor.count("face_floor.rho",
                  count_below(lr[side][IDN],
                              hydro.peos->options->density_floor(), dim - 1));
    lr[side][IPR].clamp_min_(hydro.peos->options->pressure_floor());
    lr[side][IDN].clamp_min_(hydro.peos->options->density_floor());
  }
  return lr;
}

void riemann_solve(HydroImpl& hydro, torch::Tensor lr, int dim,
                   torch::Tensor flux, torch::Tensor face_pressure,
                   torch::Tensor& background,
                   IdealMoistPerturbations const* perturb,
                   torch::Tensor central) {
  // the solve's events (C5): hllc's clamped contact pressure, roe's LLF
  auto post_events = [&] {
    if (!hydro.priemann->events.defined()) return;
    hydro.flux_register().monitor().count("riemann.pstar_clamp",
                                          hydro.priemann->events[0]);
    hydro.flux_register().monitor().count("riemann.llf",
                                          hydro.priemann->events[1]);
  };
  if (auto g = hydro.pmb->eb_geometry(); g && g->panel >= 0) {
    sphere_riemann(hydro, lr, dim, flux, face_pressure, central);
    post_events();
    return;
  }
  if (!perturb) {
    hydro.priemann->forward(lr[ILT], lr[IRT], dim, flux, face_pressure,
                            central);
    post_events();
    return;
  }
  if (!background.defined()) {
    // Horizontal (or gravity-free) sweeps: the shared face gauge need not
    // be hydrostatic. Its full gradient is retained in the momentum source.
    // A static gauge keeps physical pressure increments in the Riemann
    // flux instead of moving them into a difference of large backgrounds.
    int axis = dim - 1;
    auto p0 = perturb->pressure0;
    int n = p0.size(axis);
    auto below =
        torch::cat({p0.narrow(axis, 0, 1), p0.narrow(axis, 0, n - 1)}, axis);
    background = 0.5 * (p0 + below);
    lr = lr.clone();
    lr[ILT][IPR] -= background;
    lr[IRT][IPR] -= background;
  }
  if (hydro.fp32()) lr = lr.to(torch::kFloat32);
  hydro.priemann->forward_perturbations(lr[ILT], lr[IRT], dim, flux, background,
                                        face_pressure, central);
  post_events();
  // Public face pressure and all implicit/energy consumers keep TOTALS.
  if (face_pressure.defined() && face_pressure.numel())
    face_pressure += background;
}

void solve_faces(HydroImpl& hydro, FaceDir const& dir,
                 torch::Tensor const& faces,
                 IdealMoistPerturbations const* perturb,
                 torch::Tensor& background, bool central,
                 FaceHook const& before_solve) {
  auto wlr = faces;
  if (dir.disabled) return;
  if (before_solve) before_solve(wlr, dir.dim);
  int d = 4 - dir.dim;
  auto& reg = hydro.flux_register();
  // riemann.central@d, kept for A until the next stage (R-2, C5)
  riemann_solve(
      hydro, wlr, dir.dim, reg.slot(d, "riemann"), torch::Tensor(), background,
      perturb, central ? reg.part_slot(d, "riemann.central") : torch::Tensor());
  reg.post_slot(d, "riemann");
}

}  // namespace snap
