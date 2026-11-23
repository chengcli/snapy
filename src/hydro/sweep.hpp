#pragma once

// C/C++
#include <functional>
#include <map>
#include <string>

// torch
#include <torch/torch.h>

// snap
#include <snap/snap.h>

namespace snap {

class HydroImpl;
class MeshBlockImpl;
struct EBGeometry;
struct CutCellStencil;
class SphereReconstruction;

// Linear reconstruction on physically connected volume centroids. Only faces
// whose ordinary stencil intersects the embedded wall use this reconstruction.
class CutCellReconstruction {
 public:
  CutCellReconstruction(EBGeometry const& geometry, torch::Tensor const& state,
                        int width, const MeshBlockImpl* sphere_block = nullptr,
                        bool velocity = false);
  torch::Tensor faces(torch::Tensor const& regular, int dim, int width) const;
  torch::Tensor wall_state() const;
  torch::Tensor irregular_faces(int dim, int width) const;

 private:
  torch::Tensor evaluate(torch::Tensor const& points) const;
  EBGeometry const& _geometry;
  std::shared_ptr<SphereReconstruction> _mapped;
  std::shared_ptr<CutCellStencil> _stencil;
  torch::Tensor _state, _packed, _gradient, _lower, _upper;
};
class ReconstructImpl;
class RegisterMonitor;
struct IdealMoistPerturbations;
using Variables = std::map<std::string, torch::Tensor>;

//! what differs between the face sweeps; everything else is shared
struct FaceDir {
  int dim;        // DIM1 = 3, DIM2 = 2, DIM3 = 1
  int seam_tag;   // x1_seam_rows tags 0x7730 (x2), 0x7734 (x3)
  bool disabled;  // disable-flux-x<n>
};

//! FaceFloor (C.rst 3.7): the faces the reconstruction's floors changed
void post_face_floors(ReconstructImpl const& recon, RegisterMonitor& monitor);

//! the face states [wl, wr] of w along dir. perturb: the split path's
//! background and increments (null on the total path); the FP32 split path
//! reconstructs the gauge perturbation, floors the totals itself and, on an
//! active horizontal sweep, sets background, its static face gauge
torch::Tensor reconstruct_faces(HydroImpl& hydro, FaceDir const& dir,
                                torch::Tensor const& w,
                                IdealMoistPerturbations const* perturb,
                                torch::Tensor& background);

//! one Riemann solve of the face states lr into flux, with its events. On the
//! split path (perturb) the solver sees perturbations about background (the
//! static face gauge if undefined) and face_pressure keeps totals
void riemann_solve(HydroImpl& hydro, torch::Tensor lr, int dim,
                   torch::Tensor flux, torch::Tensor face_pressure,
                   torch::Tensor& background,
                   IdealMoistPerturbations const* perturb,
                   torch::Tensor central = torch::Tensor());

//! called with the unrotated face states before the solve
using FaceHook = std::function<void(torch::Tensor const&, int)>;

//! the x2/x3 sweep after reconstruction: before_solve ->
//! Riemann into slot(d, "riemann"), riemann.central@d when central, then
//! post_slot. Disabled sweeps return immediately
void solve_faces(HydroImpl& hydro, FaceDir const& dir,
                 torch::Tensor const& faces,
                 IdealMoistPerturbations const* perturb,
                 torch::Tensor& background, bool central,
                 FaceHook const& before_solve = {});

}  // namespace snap
