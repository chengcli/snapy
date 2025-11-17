#pragma once

// C/C++
#include <iostream>

// snap
#include "connectivity.hpp"

// arg
#include <snap/add_arg.h>

namespace snap {

/*!
 * \brief Calculate buffer ID from directional offsets
 *
 * Converts 3D directional offsets into a linear buffer index.
 * For 2D layouts (dz=0), returns index in range [0,8].
 * For 3D layouts, returns index in range [0,26].
 *
 * \param dx offset in x3 direction (-1, 0, or 1)
 * \param dy offset in x2 direction (-1, 0, or 1)
 * \param dz offset in x1 direction (-1, 0, or 1)
 * \return linear buffer index
 */
inline int get_buffer_id(int dx, int dy, int dz = 0) {
  return (dx % 3 + 3) % 3 + ((dy % 3 + 3) % 3) * 3 + ((dz % 3 + 3) % 3) * 9;
}

struct LayoutOptions {
  void report(std::ostream &os) const {
    os << "* type=" << type() << "\n";
    os << "* px=" << px() << "\n";
    os << "* py=" << py() << "\n";
    os << "* pz=" << pz() << "\n";
    os << "* periodic_x=" << (periodic_x() ? "true" : "false") << "\n";
    os << "* periodic_y=" << (periodic_y() ? "true" : "false") << "\n";
    os << "* periodic_z=" << (periodic_z() ? "true" : "false") << "\n";
  }

  //! type of layout
  ADD_ARG(std::string, type) = "slab";

  //! number of processors in X
  ADD_ARG(int, px) = 1;

  //! number of processors in Y
  ADD_ARG(int, py) = 1;

  //! number of processors in Z
  ADD_ARG(int, pz) = 1;

  //! periodicity in X
  ADD_ARG(bool, periodic_x) = false;

  //! periodicity in Y
  ADD_ARG(bool, periodic_y) = false;

  //! periodicity in Z
  ADD_ARG(bool, periodic_z) = false;
};

class LayoutImpl {
 public:
  //! options with which this `Layout` was constructed
  LayoutOptions options;
  LayoutImpl(const LayoutOptions &opts, int copies = 1) : options(opts) {
    int P = copies * options.px() * options.py() * options.pz();
    _rankof = new int[P];
  }

  std::tuple<int, int, int> get_procs() const {
    return {options.px(), options.py(), options.pz()};
  }

  virtual ~LayoutImpl() { delete[] _rankof; }

  virtual void report(std::ostream &os) const { options.report(os); }

  virtual int rank_of(int rx, int ry, int rz = 0) const {
    int px = options.px();
    int py = options.py();
    int pz = options.pz();
    if (rx < 0 || rx >= px || ry < 0 || ry >= py || rz < 0 || rz >= pz)
      return -1;
    return _rankof[rz * (px * py) + ry * px + rx];
  }

  virtual std::tuple<int, int, int> loc_of(int rank) const {
    return {-1, -1, -1};
  }

  //! \brief Neighbor -> Z-order rank (3D)
  /*!
   * dx,dy,dz <- {-1,0,1}. periodic flags control wrap;
   * otherwise off-domain -> -1.
   * (rx,ry,rz) are THIS rank's coords in the process grid (not Morton code).
   */
  virtual int neighbor_rank(int rx, int ry, int rz, int dx, int dy,
                            int dz = 0) const {
    return -1;
  }

 protected:
  Coord2 *_coords2 = nullptr;
  Coord3 *_coords3 = nullptr;
  int *_rankof = nullptr;
};

class SlabLayoutImpl : public LayoutImpl {
 public:
  SlabLayoutImpl(const LayoutOptions &opts) : LayoutImpl(opts) {
    if (options.pz() != 1) {
      throw std::runtime_error("SlabLayoutImpl: pz must be 1 for slab layout");
    }

    int px = options.px();
    int py = options.py();

    _coords2 = new Coord2[px * py];
    build_zorder_coords2(px, py, _coords2);
    build_rank_of2(px, py, _coords2, _rankof);
  }

  ~SlabLayoutImpl() { delete[] _coords2; }
  void report(std::ostream &os) const override;
  std::tuple<int, int, int> loc_of(int rank) const override;
  int neighbor_rank(int rx, int ry, int rz, int dx, int dy,
                    int dz = 0) const override;
};

class CubedLayoutImpl : public LayoutImpl {
 public:
  CubedLayoutImpl(const LayoutOptions &opts) : LayoutImpl(opts) {
    int px = options.px();
    int py = options.py();
    int pz = options.pz();

    _coords3 = new Coord3[px * py * pz];
    build_zorder_coords3(px, py, pz, _coords3);
    build_rank_of3(px, py, pz, _coords3, _rankof);
  }

  ~CubedLayoutImpl() { delete[] _coords3; }

  void report(std::ostream &os) const override;
  std::tuple<int, int, int> loc_of(int rank) const override;
  int neighbor_rank(int rx, int ry, int rz, int dx, int dy,
                    int dz = 0) const override;
};

class CubedSphereLayoutImpl : public LayoutImpl {
 public:
  CubedSphereLayoutImpl(const LayoutOptions &opts) : LayoutImpl(opts, 6) {
    int P = pxy() * pxy();
    _coords2 = new Coord2[6 * P];

    for (int f = 0; f < 6; ++f) {
      _coords6[f] = _coords2 + f * P;
      _rankof6[f] = _rankof + f * P;

      build_zorder_coords2(pxy(), pxy(), _coords6[f]);
      build_rank_of2(pxy(), pxy(), _coords6[f], _rankof6[f]);
    }
  }

  ~CubedSphereLayoutImpl() { delete[] _coords2; }

  int pxy() const { return options.px(); }

  int rank_of(int rx, int ry, int face) const override;
  std::tuple<int, int, int> loc_of(int global_rank) const override;

  int neighbor_rank(int rx, int ry, int face, int dx, int dy,
                    int dz = 0) const override;
  void report(std::ostream &os) const override;

 private:
  //! \brieff Global rank layout: face-major, Z-order within face
  int _global_rank_from_face_local(int face, int r_local) const {
    int P = pxy() * pxy();
    return face * P + r_local;
  }

  //! \brief Reverse: get (face, r_local) from global rank */
  void _global_rank_to_face_local(int grank, int *face, int *r_local) const {
    int P = pxy() * pxy();
    *face = grank / P;
    *r_local = grank % P;
  }

  //! \brief map local (rx,ry) to per-face Z-order rank */
  int _face_local_rank(int face, int rx, int ry) const {
    return _rankof6[face][linear_index2(pxy(), pxy(), ry, rx)];
  }

  //! \brief Edge stepping helper
  /*!
   * Move off the face by one tile in (dx,dy) ∈ {-1,0,1}^2.
   * Returns neighbor (nface, nrank) or (-1, -1) on error (should not happen on
   * a closed cube).
   *
   * Logic:
   * - If inside same face: trivial offset of (rx,ry).
   * - If crossing a single edge (|dx|+|dy|==1): use edge table to decide
   *    neighbor face & side, compute the along-edge index (pos), reverse if
   *    needed, and place at neighbor border.
   * - If crossing a corner (|dx|==1 && |dy|==1): do it in two hops.
   *    (dx,0) and (0,dy) through the intermediate face.
   *    If across a panel boundary, do first step inside the panel
   *    and second step outside. This mirrors typical ghost-corner
   *    exchange.
   */
  void _step_one(int face, int rx, int ry, int dx, int dy, int *out_face,
                 int *out_rx, int *out_ry) const;

  Coord2 *_coords6[6];  //! coords per face: length P=px*py each
  int *_rankof6[6];     //! inverse map per face: length P=px*py each
};

using Layout = std::shared_ptr<LayoutImpl>;
using SlabLayout = std::shared_ptr<SlabLayoutImpl>;
using CubedLayout = std::shared_ptr<CubedLayoutImpl>;
using CubedSphereLayout = std::shared_ptr<CubedSphereLayoutImpl>;

inline Layout create_layout(LayoutOptions const &opts) {
  if (opts.type() == "slab") {
    return std::make_shared<SlabLayoutImpl>(opts);
  } else if (opts.type() == "cubed") {
    return std::make_shared<CubedLayoutImpl>(opts);
  } else if (opts.type() == "cubed_sphere") {
    return std::make_shared<CubedSphereLayoutImpl>(opts);
  } else {
    throw std::runtime_error("layout type '" + opts.type() +
                             "' is not implemented.");
  }
}

}  // namespace snap

#undef ADD_ARG
