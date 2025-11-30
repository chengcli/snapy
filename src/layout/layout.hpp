#pragma once

// C/C++
#include <iostream>
#include <memory>
#include <tuple>

// torch
#include <torch/nn/cloneable.h>
#include <torch/nn/module.h>
#include <torch/nn/modules/common.h>

#include <torch/csrc/distributed/c10d/Store.hpp>
// #include <torch/csrc/distributed/c10d/ProcessGroup.hpp>
#include <torch/csrc/distributed/c10d/Backend.hpp>

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
inline int get_buffer_id(std::tuple<int, int, int> offset) {
  auto [dx, dy, dz] = offset;
  return (dx % 3 + 3) % 3 + ((dy % 3 + 3) % 3) * 3 + ((dz % 3 + 3) % 3) * 9;
}

//! get environment variable with default
inline std::string get_env(const char *name, const std::string &def) {
  const char *v = std::getenv(name);
  return v ? std::string(v) : def;
}

//! get global rank from environment variable
inline int get_rank() { return std::stoi(get_env("RANK", "0")); }

//! get local rank from environment variable
inline int get_local_rank() { return std::stoi(get_env("LOCAL_RANK", "0")); }

struct LayoutOptionsImpl {
  static std::shared_ptr<LayoutOptionsImpl> create() {
    return std::make_shared<LayoutOptionsImpl>();
  }
  static std::shared_ptr<LayoutOptionsImpl> from_yaml(
      std::string const &filename);

  LayoutOptionsImpl();

  void report(std::ostream &os) const {
    os << "* type=" << type() << "\n"
       << "* px=" << px() << "\n"
       << "* py=" << py() << "\n"
       << "* pz=" << pz() << "\n"
       << "* periodic_x=" << (periodic_x() ? "true" : "false") << "\n"
       << "* periodic_y=" << (periodic_y() ? "true" : "false") << "\n"
       << "* periodic_z=" << (periodic_z() ? "true" : "false") << "\n"
       << "* backend = " << backend() << "\n"
       << "* master_addr = " << master_addr() << "\n"
       << "* rank = " << rank() << "\n"
       << "* local_rank = " << local_rank() << "\n"
       << "* world_size = " << world_size() << "\n"
       << "* master_port = " << master_port() << "\n"
       << "* verbose = " << (verbose() ? "true" : "false") << "\n";
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

  ADD_ARG(std::string, backend) = "gloo";
  ADD_ARG(std::string, master_addr) = "127.0.0.1";
  ADD_ARG(int, rank) = 0;
  ADD_ARG(int, root_rank) = 0;
  ADD_ARG(int, local_rank) = 0;
  ADD_ARG(int, world_size) = 1;
  ADD_ARG(int, master_port) = 29500;
  ADD_ARG(bool, verbose) = false;
};
using LayoutOptions = std::shared_ptr<LayoutOptionsImpl>;

using Variables = std::map<std::string, torch::Tensor>;

class MeshBlockImpl;

class LayoutImpl {
 public:
  static std::shared_ptr<LayoutImpl> create(LayoutOptions const &opts,
                                            torch::nn::Module *p = nullptr,
                                            std::string const &name = "layout");

  //! exchange buffers
  /*!
   * The first index indicates the rank
   * The second index indicates the variable group
   */
  std::vector<std::vector<torch::Tensor>> send_bufs, recv_bufs;

  //! buffer variable names
  std::vector<std::string> buf_names;

  //! submodules
  at::intrusive_ptr<c10d::Store> store;
  std::shared_ptr<c10d::Backend> pg;

  //! options with which this `Layout` was constructed
  LayoutOptions options;

  LayoutImpl() : options(LayoutOptionsImpl::create()) {}
  LayoutImpl(const LayoutOptions &opts, int copies = 1) : options(opts) {
    int P = copies * options->px() * options->py() * options->pz();
    _rankof.resize(P);
  }

  std::tuple<int, int, int> get_procs() const {
    return {options->px(), options->py(), options->pz()};
  }

  bool is_root() const { return options->rank() == options->root_rank(); }

  virtual ~LayoutImpl() = default;

  virtual int rank_of(std::tuple<int, int, int> iloc) const {
    auto [rx, ry, rz] = iloc;

    int px = options->px();
    int py = options->py();
    int pz = options->pz();
    if (rx < 0 || rx >= px || ry < 0 || ry >= py || rz < 0 || rz >= pz)
      return -1;
    return _rankof[rz * (px * py) + ry * px + rx];
  }

  virtual std::tuple<int, int, int> loc_of(int rank) const = 0;

  //! \brief Neighbor -> Z-order rank (3D)
  /*!
   * offset = (dx,dy,dz) <- {-1,0,1}. periodic flags control wrap;
   * otherwise off-domain -> -1.
   * iloc = (rx,ry,rz) are THIS rank's logical coords in the process grid (not
   * Morton code).
   */
  virtual int neighbor_rank(std::tuple<int, int, int> iloc,
                            std::tuple<int, int, int> offset) const = 0;

  //! \brief Initialize send and receive buffers for 2D domain decomposition
  /*!
   * Allocates torch::Tensor buffers for exchanging ghost zone data with
   * neighboring processes in a 2D slab decomposition. Buffers are sized
   * to match the ghost zone dimensions of the mesh block.
   */
  virtual void init_buffers(MeshBlockImpl const *pmb, Variables const &vars,
                            std::vector<std::string> const &names);

  //! Serialize variables
  virtual void serialize(MeshBlockImpl const *pmb, Variables const &vars);

  //! Deserialize variables
  virtual void deserialize(MeshBlockImpl const *pmb, Variables &vars) const;

  //! \brief Perform ghost zone exchange
  /*!
   * Exchanges ghost zone data with neighboring processes using point-to-point
   * communication. This function serializes data, performs send/recv
   * operations, and deserializes received data into ghost zones.
   */
  virtual void forward(MeshBlockImpl const *pmb, Variables &vars) {}

 protected:
  void _init_backend();
  // --- Backend initializers ---
  void _init_gloo();
  void _init_nccl();

  std::vector<Coord2> _coords2;
  std::vector<Coord3> _coords3;
  std::vector<int> _rankof;
};
using Layout = std::shared_ptr<LayoutImpl>;

class SlabLayoutImpl : public torch::nn::Cloneable<SlabLayoutImpl>,
                       public LayoutImpl {
 public:
  //! Constructor to initialize the layers
  SlabLayoutImpl() = default;
  SlabLayoutImpl(const LayoutOptions &opts) : LayoutImpl(opts) { reset(); }
  void reset() override;

  ~SlabLayoutImpl() = default;
  void pretty_print(std::ostream &os) const override;

  std::tuple<int, int, int> loc_of(int rank) const override;
  int neighbor_rank(std::tuple<int, int, int> iloc,
                    std::tuple<int, int, int> offset) const override;

  //! \brief Perform ghost zone exchange for slab layout
  void forward(MeshBlockImpl const *pmb, Variables &vars) override;
};
TORCH_MODULE(SlabLayout);

class CubedLayoutImpl : public torch::nn::Cloneable<CubedLayoutImpl>,
                        public LayoutImpl {
 public:
  //! Constructor to initialize the layers
  CubedLayoutImpl() = default;
  CubedLayoutImpl(const LayoutOptions &opts) : LayoutImpl(opts) { reset(); }
  void reset() override;

  ~CubedLayoutImpl() = default;
  void pretty_print(std::ostream &os) const override;

  std::tuple<int, int, int> loc_of(int rank) const override;
  int neighbor_rank(std::tuple<int, int, int> iloc,
                    std::tuple<int, int, int> offset) const override;
};
TORCH_MODULE(CubedLayout);

class CubedSphereLayoutImpl
    : public torch::nn::Cloneable<CubedSphereLayoutImpl>,
      public LayoutImpl {
 public:
  //! Constructor to initialize the layers
  CubedSphereLayoutImpl() = default;
  CubedSphereLayoutImpl(const LayoutOptions &opts) : LayoutImpl(opts, 6) {
    reset();
  }
  void reset() override;

  ~CubedSphereLayoutImpl() = default;
  void pretty_print(std::ostream &os) const override;

  int pxy() const { return options->px(); }

  int rank_of(std::tuple<int, int, int> iloc) const override;
  std::tuple<int, int, int> loc_of(int global_rank) const override;

  int neighbor_rank(std::tuple<int, int, int> iloc,
                    std::tuple<int, int, int> offset) const override;

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
TORCH_MODULE(CubedSphereLayout);

}  // namespace snap

#undef ADD_ARG
