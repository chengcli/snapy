#pragma once

// C/C++
#include <vector>

// torch
#include <torch/torch.h>

namespace snap {

class FluxRegister;
class MeshBlockImpl;

//! V3 lane S, step (4) (S.rst rev 2.5 sections 4 and 10, phase 3): the
//! shared faces S hands to R's register.

//! Row 16r, the cubed-sphere panel seams: the block with the smaller global
//! rank owns a seam face (SeamOwner::lower_rank). The owner sends its face
//! rows, the total's and then each kept part's in kept_parts(dim) order
//! (R-4), with the momentum rows taken from its covariant face basis to the
//! global Cartesian frame. The non-owner maps them by the panel edge (along
//! the edge, reversed if CS_FACE_EDGES says so; every row times -1 where the
//! two normals are opposite), takes the momentum rows into its own covariant
//! face basis and calls reg.replace(dim, faces, peer, SeamRule::owner_copy)
//! once per direction. Intra-panel edges are left alone (SeamOwner::self).
//! A no-op off the cubed sphere. Messages carry tag + the receiver's side.
void reconcile_panel_seams(FluxRegister& reg, MeshBlockImpl const* pmb,
                           int tag);

//! boundary.prescribed: S's mask for R's set_prescribed, per active
//! direction, on every register instance (the hydro's and each Scalar
//! module's). Physical wall fluxes are zero except for free_rows.
void supply_prescribed(FluxRegister& reg, MeshBlockImpl const* pmb,
                       std::vector<int> const& free_rows);

}  // namespace snap
