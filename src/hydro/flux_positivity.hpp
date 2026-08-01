#pragma once

// torch
#include <torch/torch.h>

// snap
#include <snap/coord/coordinate.hpp>

namespace snap {

//! \brief Per-cell positivity limiter factors for donor-form tracer fluxes.
//!
//! For each channel c and cell i, sums the outgoing (donor-side) flux over all
//! faces of the cell exactly as the divergence will apply them,
//!   out_i = sum_faces max(+/- A*F, 0),
//! and returns
//!   theta_i = min(1, u_i * V_i / (dt * out_i)),
//! the largest uniform scaling of cell i's outgoing fluxes that cannot drain
//! the cell below zero in one forward-Euler step of size dt. Applying theta of
//! the donor cell to every face (flux_positivity_scale_) then guarantees
//!   u_i + dt * du_i(transport) >= 0
//! cell by cell. All shipped integrators (rk1/rk2/rk3: wght2 == wght1 per
//! stage; rk3s4: wght2 <= wght1) form each stage as a convex combination of
//! previous states and one full-dt Euler step, so per-stage limiting at the
//! full dt preserves non-negativity of the stage updates as well.
//!
//! theta == 1 wherever the cell is not near depletion, so the high-order flux
//! is untouched almost everywhere; conservation is exact because each face is
//! scaled by a single factor shared by both adjacent cells.
//!
//! Ghost cells get theta = 1 here (their outflow sum is not computed); the
//! caller must make donor factors single-valued at internal seams by filling
//! theta's ghost layer the same way conserved-variable ghosts are filled
//! (exchange + physical boundary functions) BEFORE calling
//! flux_positivity_scale_.
//!
//! \param u      conserved tracer densities, (nchan, nc3, nc2, nc1)
//! \param flux1/2/3  channel-sliced flux views, same layout as u; undefined
//!               tensors are skipped. flux[i] is the flux through the lower
//!               face of cell i (divergence convention).
//! \param pcoord coordinate providing face_area1/2/3 and cell_volume
//! \param dt     full stage time step
torch::Tensor flux_positivity_theta(torch::Tensor const& u,
                                    torch::Tensor const& flux1,
                                    torch::Tensor const& flux2,
                                    torch::Tensor const& flux3,
                                    Coordinate const& pcoord, double dt);

//! \brief Scale each face's flux by the donor cell's theta, in place.
//!
//! The donor of a face is the cell the flux drains: the lower cell where the
//! (per-channel) flux is positive, the upper cell otherwise. Only the faces
//! the divergence consumes (lower faces il..iu+1 per dimension) are touched.
void flux_positivity_scale_(torch::Tensor const& theta,
                            torch::Tensor const& flux1,
                            torch::Tensor const& flux2,
                            torch::Tensor const& flux3,
                            Coordinate const& pcoord);

}  // namespace snap
