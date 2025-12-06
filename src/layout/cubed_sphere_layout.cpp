//! \brief Cubed-sphere layout implementation
/*!
 * Notes on connectivity:
 *  - We model only the surface (six 2D faces).
 *  - Each face holds a px-by-py processor grid (px==py required by the cubed
 * grid; however code allows px,py).
 *  - Global rank = face_number * (py*px) + zorder_rank_within_face.
 *
 * Orientation model across an edge:
 *   From (face f, side s \in {L,R,B,T}) we land on (nface, nside),
 *   and the along-edge index is either preserved or reversed.
 *   No "transpose" is required if nside is defined correctly:
 *     - neighbor side L/R varies along neighbor Y (rows)
 *     - neighbor side B/T varies along neighbor X (cols)
 *
 * Face naming:
 *   Faces are named according to the global carteian +X,+Y,+Z,-X,-Y,-Z
 * directions Face names should be consistent with the numbering of faces in the
 * global cartesian coordinates.
 *
 *   Table `CS_FACE_EDGES[6][4]` and `CS_FACE_NAMES[6][3]` should be
 *   edited together to ensure consistency.
 *
 * Ghost zone communication:
 *   The main function to perform ghost zone communication is
 *    CubedSphereLayoutImpl::forward().
 *   The loc_of() function return a 3-item tuple (rx,ry,face) for a given rank.
 *   Position of a serialization/deserialization buffer is identified by
 *   get_buffer_id(offset), where offset is a 3-item tuple (dx,dy,0)
 *
 *   CubedSphereLayoutImpl::serialize() saves the ghost zone data into send
 * buffers. CubedSphereLayoutImpl::deserialize() loads the received data into
 * ghost zones.
 *
 *   For the cubed-sphere layout, only face-adjacent neighbors are considered.
 *   Within serialization, _covariant_to_cartesian() is called to convert
 *   covariant vector components to Cartesian components.
 *
 *   Within deserialization, _cartesian_to_covariant() is called to convert
 *   Cartesian vector components back to covariant components.
 *
 *   These functions will use the coordinate information from the MeshBlock.
 *   Access the coordinate via `pmb->pcoord`.
 *
 *   The cell-centered angular coordinates are in `pcoord->x2v` and
 * `pcoord->x3v`. Consult src/coord/coordinate.hpp,
 * src/coord/gnomonic_equiangular.cpp, for more details.
 */

// C/C++
#include <cstdint>
#include <cstdio>
#include <cstdlib>

// fmt
#include <fmt/format.h>

// snap
#include <snap/snap.h>

#include <snap/coord/coordinate.hpp>
#include <snap/mesh/meshblock.hpp>

#include "connectivity.hpp"
#include "cubed_sphere_layout.hpp"

namespace snap {

/*!
 * ----------------------------
 * Global cartesian coordiantes
 * ----------------------------
 *
 *       +Z
 *       ^
 *       |
 *       |----> +Y
 *      /
 *  +X /
 *
 * --------------------------
 * Cubed-sphere connectivity
 * --------------------------
 * Face numbering (editable):
 *
 *           -------
 *           |  3  |
 *     |-----|-----|-----|-----|
 *     |  4  |  0  |  1  |  2  |
 *     |-----|-----|-----|-----|
 *           |  5  |
 *           |-----|
 *
 * Sides: 0=L, 1=R, 2=B, 3=T  (left, right, bottom, top)
 * Each entry says: leaving face F via side S,
 * you arrive at (nface, nside) and the along-edge index is reversed? (0/1)
 *
 * -------------------------------
 * Local face orientaion and sides
 * -------------------------------
 *
 *         (T,3)          beta
 *        |-----|         ^
 *  (L,0) |  X  | (R,1)   |
 *        |-----|         |----> alpha
 *         (B,2)
 *
 * IMPORTANT: Different codes choose different local face axes.
 * If your tests show flipped corner order, toggle `rev` for that edge.
 */

// face 0: +X
// face 1: +Y
// face 2: -X
// face 3: +Z
// face 4: -Y
// face 5: -Z
const char CS_FACE_NAMES[6][3] = {"+X", "+Y", "-X", "+Z", "-Y", "-Z"};

const CSEdge CS_FACE_EDGES[6][4] = {
    /* face 0: neighbors 4(L),1(R),5(B),3(T) */
    [0] = {/* L */ {4, SIDE_R, 0},
           /* R */ {1, SIDE_L, 0},
           /* B */ {5, SIDE_T, 0},
           /* T */ {3, SIDE_B, 0}},
    /* face 1: neighbors 0(L),2(R),5(B),3(T) */
    [1] = {/* L */ {0, SIDE_R, 0},
           /* R */ {2, SIDE_L, 0},
           /* B */ {5, SIDE_R, 1},
           /* T */ {3, SIDE_R, 0}},
    /* face 2: neighbors 1(L),4(R),5(B),3(T) */
    [2] = {/* L */ {1, SIDE_R, 0},
           /* R */ {4, SIDE_L, 0},
           /* B */ {5, SIDE_B, 1},
           /* T */ {3, SIDE_T, 1}},
    /* face 3: neighbors 4(L),1(R),0(B),2(T) */
    [3] = {/* L */ {4, SIDE_T, 1},
           /* R */ {1, SIDE_T, 0},
           /* B */ {0, SIDE_T, 0},
           /* T */ {2, SIDE_T, 1}},
    /* face 4: neighbors 2(L),0(R),5(B),3(T) */
    [4] = {/* L */ {2, SIDE_R, 0},
           /* R */ {0, SIDE_L, 0},
           /* B */ {5, SIDE_L, 0},
           /* T */ {3, SIDE_L, 1}},
    /* face 5: neighbors 4(L),1(R),2(B),0(T) */
    [5] = {/* L */ {4, SIDE_B, 0},
           /* R */ {1, SIDE_B, 1},
           /* B */ {2, SIDE_B, 1},
           /* T */ {0, SIDE_B, 0}}};

static inline void cs_clamp_inside(int pxy, int *nx, int *ny) {
  if (*nx < 0)
    *nx = 0;
  else if (*nx >= pxy)
    *nx = pxy - 1;
  if (*ny < 0)
    *ny = 0;
  else if (*ny >= pxy)
    *ny = pxy - 1;
}

static inline void cs_edge_map_into_neighbor(int pxy, int leaving_side,
                                             int pos /*0..k-1*/,
                                             const CSEdge *emap, int *out_rx,
                                             int *out_ry) {
  /* Map along-edge index into neighbor face border, with optional reversal. */
  int pos2 = pos;
  if (emap->rev) {
    pos2 = pxy - 1 - pos;
  }

  switch (emap->nside) {
    case SIDE_L:
      *out_rx = 0;
      *out_ry = pos2;
      break; /* varies in y */
    case SIDE_R:
      *out_rx = pxy - 1;
      *out_ry = pos2;
      break;
    case SIDE_B:
      *out_rx = pos2;
      *out_ry = 0;
      break; /* varies in x */
    case SIDE_T:
      *out_rx = pos2;
      *out_ry = pxy - 1;
      break;
    default:
      *out_rx = 0;
      *out_ry = 0;
      break;
  }
}

void CubedSphereLayoutImpl::reset() {
  // build the ranks
  TORCH_CHECK(options->pz() == 1,
              "CubedSphereLayoutImpl: pz must be 1 for cubed-sphere layout");
  TORCH_CHECK(
      options->px() == options->py(),
      "CubedSphereLayoutImpl: px must equal py for cubed-sphere layout");

  int px = options->px();
  int py = options->py();

  _coords2.resize(px * py);
  build_zorder_coords2(pxy(), pxy(), _coords2.data());
  build_rank_of2(pxy(), pxy(), _coords2.data(), _rankof.data());

  // build backend
  _init_backend();
}

void CubedSphereLayoutImpl::pretty_print(std::ostream &os) const {
  options->report(os);
  for (int f = 0; f < 6; ++f) {
    os << " Rank | (rx,ry;f)\n";
    os << "----------------\n";
    for (int r = 0; r < pxy() * pxy(); ++r) {
      int gr = _global_rank_from_face_local(f, r);
      os << fmt::format(" {:>3} | ({:>2},{:>2};{:>2})\n", gr, _coords2[r].x,
                        _coords2[r].y, f);
    }
  }
}

void CubedSphereLayoutImpl::_step_one(int face, int rx, int ry, int dx, int dy,
                                      int *out_face, int *out_rx,
                                      int *out_ry) const {
  /* Try to stay on-face */
  int nx = rx + dx;
  int ny = ry + dy;
  if (0 <= nx && nx < pxy() && 0 <= ny && ny < pxy()) {
    *out_face = face;
    *out_rx = nx;
    *out_ry = ny;
    return;
  }

  /* Identify which single edge is crossed */
  int side = -1;
  if (nx < 0)
    side = SIDE_L;
  else if (nx >= pxy())
    side = SIDE_R;
  else if (ny < 0)
    side = SIDE_B;
  else if (ny >= pxy())
    side = SIDE_T;

  const CSEdge emap = CS_FACE_EDGES[face][side];
  *out_face = emap.nface;

  /* Along-edge position on current face */
  int pos = (side == SIDE_L || side == SIDE_R) ? ry : rx;

  /* Map to neighbor border */
  cs_edge_map_into_neighbor(pxy(), side, pos, &emap, out_rx, out_ry);

  // cs_clamp_inside(pxy, out_rx, out_ry);
}

int CubedSphereLayoutImpl::rank_of(std::tuple<int, int, int> iloc) const {
  auto [rx, ry, face] = iloc;
  if (face < 0 || face >= 6) return -1;
  if (rx < 0 || rx >= pxy() || ry < 0 || ry >= pxy()) return -1;
  return _rankof[ry * pxy() + rx];
}

std::tuple<int, int, int> CubedSphereLayoutImpl::loc_of(int global_rank) const {
  if (global_rank < 0 || global_rank >= 6 * pxy() * pxy()) return {-1, -1, -1};
  int face, r_local;
  _global_rank_to_face_local(global_rank, &face, &r_local);
  int rx = _coords2[r_local].x;
  int ry = _coords2[r_local].y;
  return {rx, ry, face};
}

/* get neighbor GLOBAL rank for (dx,dy) in {-1,0,1}^2 (incl. corners) */
int CubedSphereLayoutImpl::neighbor_rank(
    std::tuple<int, int, int> iloc, std::tuple<int, int, int> offset) const {
  auto [rx, ry, face] = iloc;
  auto [dx, dy, _] = offset;

  if (dx == 0 && dy == 0) {
    /* self */
    int rloc = _face_local_rank(rx, ry);
    return _global_rank_from_face_local(face, rloc);
  }

  /* 1-step edge move */
  if ((dx == 0) ^ (dy == 0)) {
    int f1, x1, y1;
    _step_one(face, rx, ry, dx, dy, &f1, &x1, &y1);
    int rloc = _face_local_rank(x1, y1);
    return _global_rank_from_face_local(f1, rloc);
  }

  /* corners: at least crossing one edge, maybe two */
  // find the current block's logical location
  int lx, ly;
  logical_loc2(rx, ry, pxy(), pxy(), &lx, &ly);

  if ((dx + lx <= 1) && (dx + lx >= -1)) {
    // do (dx,0) and then (0,dy)
    // printf("lx = %d, ly = %d, dx = %d, dy = %d\n", lx, ly, dx, dy);
    int f1, x1, y1;
    _step_one(face, rx, ry, dx, 0, &f1, &x1, &y1);

    int f2, x2, y2;
    _step_one(f1, x1, y1, 0, dy, &f2, &x2, &y2);
    int rloc = _face_local_rank(x2, y2);
    return _global_rank_from_face_local(f2, rloc);
  } else if ((dy + ly <= 1) && (dy + ly >= -1)) {
    // do (0, dy) and then (dx, 0)
    int f1, x1, y1;
    _step_one(face, rx, ry, 0, dy, &f1, &x1, &y1);

    int f2, x2, y2;
    _step_one(f1, x1, y1, dx, 0, &f2, &x2, &y2);
    int rloc = _face_local_rank(x2, y2);
    return _global_rank_from_face_local(f2, rloc);
  } else {  // crossing two edges
    int f1, x1, y1;
    _step_one(face, rx, ry, dx, 0, &f1, &x1, &y1);
    int rloc = _face_local_rank(x1, y1);
    return _global_rank_from_face_local(f1, rloc);
  }
}

void CubedSphereLayoutImpl::forward(MeshBlockImpl const *pmb, Variables &vars,
                                    SyncOptions opts) {
  TORCH_CHECK(!options->no_backend(),
              "[CubedSphereLayout:forward] backend is disabled");
  TORCH_CHECK(pmb != nullptr,
              "[CubedSphereLayout:forward] MeshBlock pointer is null");

  // Skip corner exchanges for cubed-sphere layout
  opts.skip_corner(true);

  // Serialize data into send buffers
  serialize(pmb, vars, opts);

  if (options->verbose() && is_root()) {
    std::cout << "[CubedSphereLayout] performing communication\n";
  }

  std::vector<c10::intrusive_ptr<c10d::Work>> works;

  // Get my rank
  auto rank = options->rank();

  // Get my logical location
  auto iloc = loc_of(rank);

  int x3_omin = opts.x3_offset_min();
  int x3_omax = opts.x3_offset_max();
  int x2_omin = opts.x2_offset_min();
  int x2_omax = opts.x2_offset_max();

  for (int x3_offset = x3_omin; x3_offset <= x3_omax; ++x3_offset)
    for (int x2_offset = x2_omin; x2_offset <= x2_omax; ++x2_offset) {
      // skip the center (self)
      if (x3_offset == 0 && x2_offset == 0) continue;
      if (opts.skip_corner() && std::abs(x3_offset) + std::abs(x2_offset) == 2)
        continue;

      std::tuple<int, int, int> offset(x3_offset, x2_offset, 0);
      int nb = neighbor_rank(iloc, offset);
      if (nb < 0) continue;  // no neighbor

      int r = get_buffer_id(offset);

      if (nb != rank) {  // different ranks
        // Send operation
        auto send_work = pg->send(send_bufs[r], nb, 0);
        works.push_back(send_work);

        // Receive operation
        auto recv_work = pg->recv(recv_bufs[r], nb, 0);
        works.push_back(recv_work);
      } else {  // self-send
        for (int n = 0; n < recv_bufs[r].size(); ++n)
          recv_bufs[r][n].copy_(send_bufs[r][n]);
      }
    }

  // Wait for all operations to complete
  for (auto &work : works) work->wait();

  // Deserialize received data into ghost zones
  deserialize(pmb, vars, opts);
}

void CubedSphereLayoutImpl::serialize(MeshBlockImpl const *pmb, Variables &vars,
                                      SyncOptions opts) {
  if (options->verbose() && is_root()) {
    std::cout << "[CubedSphereLayout] serializing data into send buffers\n";
  }

  // Get my logical location
  auto iloc = loc_of(options->rank());

  // Iterate over all face-adjacent neighbor directions
  int x3_omin = opts.x3_offset_min();
  int x3_omax = opts.x3_offset_max();
  int x2_omin = opts.x2_offset_min();
  int x2_omax = opts.x2_offset_max();

  for (int x3_offset = x3_omin; x3_offset <= x3_omax; ++x3_offset)
    for (int x2_offset = x2_omin; x2_offset <= x2_omax; ++x2_offset) {
      // skip the center (self)
      if (x3_offset == 0 && x2_offset == 0) continue;
      if (opts.skip_corner() && std::abs(x3_offset) + std::abs(x2_offset) == 2)
        continue;

      std::tuple<int, int, int> offset(x3_offset, x2_offset, 0);
      int nb = neighbor_rank(iloc, offset);
      if (nb < 0) continue;  // no neighbor

      bool inter_panel = std::get<2>(iloc) != std::get<2>(loc_of(nb));
      if (opts.cross_panel_only() && inter_panel) continue;

      // Get the interior part for this direction
      auto sub = pmb->part(offset, /*exterior=*/false);

      // Copy data from mesh to send buffer
      int bid = get_buffer_id(offset);
      send_bufs[bid].resize(vars.size());
      recv_bufs[bid].resize(vars.size());
      int count = 0;
      for (auto &[name, vara] : vars) {
        auto var = vara.index(sub);
        if (name == "hydro_u" && inter_panel) {
          _covariant_to_cartesian(pmb, offset, var[IVX], var[IVY], var[IVZ]);
        }
        send_bufs[bid][count] = var.clone();
        recv_bufs[bid][count] = torch::empty_like(send_bufs[bid][count]);
        count++;
      }
    }
}

void CubedSphereLayoutImpl::deserialize(MeshBlockImpl const *pmb,
                                        Variables &vars,
                                        SyncOptions opts) const {
  if (options->verbose() && is_root()) {
    std::cout
        << "[CubedSphereLayout] deserializing data from receive buffers\n";
  }

  // Get my logical location
  auto iloc = loc_of(options->rank());

  int x3_omin = opts.x3_offset_min();
  int x3_omax = opts.x3_offset_max();
  int x2_omin = opts.x2_offset_min();
  int x2_omax = opts.x2_offset_max();

  // Deserialize over all intra-panel neighbors first
  for (int x3_offset = x3_omin; x3_offset <= x3_omax; ++x3_offset)
    for (int x2_offset = x2_omin; x2_offset <= x2_omax; ++x2_offset) {
      // skip the center (self)
      if (x3_offset == 0 && x2_offset == 0) continue;
      if (opts.skip_corner() && std::abs(x3_offset) + std::abs(x2_offset) == 2)
        continue;

      std::tuple<int, int, int> offset(x3_offset, x2_offset, 0);
      int nb = neighbor_rank(iloc, offset);
      if (nb < 0) continue;  // no neighbor

      bool inter_panel = std::get<2>(iloc) != std::get<2>(loc_of(nb));
      if (opts.cross_panel_only() && inter_panel) continue;

      // Get the exterior (ghost zone) part for this direction
      auto sub = pmb->part(offset, /*exterior=*/true);

      // Copy data from receive buffer to mesh ghost zones
      int bid = get_buffer_id(offset);
      int count = 0;
      for (auto &[name, var] : vars) {
        var.index_put_(sub, recv_bufs[bid][count++]);
      }
    }

  // Deserialize over all inter-panel neighbors
  for (int x3_offset = x3_omin; x3_offset <= x3_omax; ++x3_offset)
    for (int x2_offset = x2_omin; x2_offset <= x2_omax; ++x2_offset) {
      // skip the center (self)
      if (x3_offset == 0 && x2_offset == 0) continue;
      // skip the corners for cubed-sphere
      if (std::abs(x3_offset) + std::abs(x2_offset) == 2) continue;

      std::tuple<int, int, int> offset(x3_offset, x2_offset, 0);
      int nb = neighbor_rank(iloc, offset);
      if (nb < 0) continue;  // no neighbor

      // skip intra-panel neighbors
      if (std::get<2>(iloc) == std::get<2>(loc_of(nb))) continue;

      // Get the exterior (ghost zone) part for this direction
      auto sub = pmb->part(offset, /*exterior=*/true);

      // Copy data from receive buffer to mesh ghost zones
      int bid = get_buffer_id(offset);
      int count = 0;
      for (auto &[name, vara] : vars) {
        vara.index_put_(sub, recv_bufs[bid][count++]);
        auto var = vara.index(sub);
        _interpolate_to_local(pmb, offset, var);
        if (name == "hydro_u") {
          _cartesian_to_covariant(pmb, offset, var[IVX], var[IVY], var[IVZ]);
        }
      }
    }
}

void CubedSphereLayoutImpl::_covariant_to_cartesian(
    MeshBlockImpl const *pmb, std::tuple<int, int, int> offset,
    torch::Tensor vz, torch::Tensor vx, torch::Tensor vy) const {
  // coordinates
  auto pcoord = pmb->phydro->pcoord;
  auto mesh = torch::meshgrid({pcoord->x3v, pcoord->x2v, pcoord->x1v},
                              /*indexing=*/"ij");

  auto sub = pmb->part(offset, /*exterior=*/false);

  auto x2v = mesh[1].unsqueeze(0).index(sub).squeeze(0);
  auto x3v = mesh[0].unsqueeze(0).index(sub).squeeze(0);

  if (options->verbose() && is_root()) {
    std::cout << "offset = (" << std::get<0>(offset) << ", "
              << std::get<1>(offset) << ", " << std::get<2>(offset) << ")\n";
    std::cout << "x2v = \n" << x2v.squeeze(-1) << "\n";
    std::cout << "x3v = \n" << x3v.squeeze(-1) << "\n";
  }

  auto co_vz = vz.clone();
  auto co_vx = vx.clone();
  auto co_vy = vy.clone();

  //\TODO transform (co_vx, co_vy, co_vz) from covariant to cartesian
}

void CubedSphereLayoutImpl::_interpolate_to_local(
    MeshBlockImpl const *pmb, std::tuple<int, int, int> offset,
    torch::Tensor var) const {
  // my coordinates
  auto pcoord = pmb->phydro->pcoord;
  auto mesh = torch::meshgrid({pcoord->x3v, pcoord->x2v, pcoord->x1v},
                              /*indexing=*/"ij");

  auto sub = pmb->part(offset, /*exterior=*/true);

  auto x2v = mesh[1].unsqueeze(0).index(sub).squeeze(0);
  auto x3v = mesh[0].unsqueeze(0).index(sub).squeeze(0);

  auto var_neighbor = var.clone();

  if (options->verbose() && is_root()) {
    std::cout << "offset = (" << std::get<0>(offset) << ", "
              << std::get<1>(offset) << ", " << std::get<2>(offset) << ")\n";
    std::cout << "var from neighbor = \n"
              << var_neighbor[IDN].squeeze().transpose(0, 1).flip(0) << "\n";
  }

  //\TODO calculate neighbor coordinates and perform interpolation
}

void CubedSphereLayoutImpl::_cartesian_to_covariant(
    MeshBlockImpl const *pmb, std::tuple<int, int, int> offset,
    torch::Tensor vz, torch::Tensor vx, torch::Tensor vy) const {
  // coordinates
  auto pcoord = pmb->phydro->pcoord;
  auto mesh = torch::meshgrid({pcoord->x3v, pcoord->x2v, pcoord->x1v},
                              /*indexing=*/"ij");

  auto sub = pmb->part(offset, /*exterior=*/true);

  auto x2v = mesh[1].unsqueeze(0).index(sub).squeeze(0);
  auto x3v = mesh[0].unsqueeze(0).index(sub).squeeze(0);

  if (options->verbose() && is_root()) {
    std::cout << "offset = (" << std::get<0>(offset) << ", "
              << std::get<1>(offset) << ", " << std::get<2>(offset) << ")\n";
    std::cout << "x2v = \n" << x2v.squeeze(-1) << "\n";
    std::cout << "x3v = \n" << x3v.squeeze(-1) << "\n";
  }

  auto cart_vz = vz.clone();
  auto cart_vx = vx.clone();
  auto cart_vy = vy.clone();

  //\TODO transform (cart_vx, cart_vy, cart_vz) from cartesian to covariant
}

}  // namespace snap
