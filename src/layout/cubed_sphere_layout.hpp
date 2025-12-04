// snap
#include "layout.hpp"

namespace snap {

class CubedSphereLayoutImpl
    : public torch::nn::Cloneable<CubedSphereLayoutImpl>,
      public LayoutImpl {
 public:
  //! Constructor to initialize the layers
  CubedSphereLayoutImpl() = default;
  CubedSphereLayoutImpl(const LayoutOptions &opts) : LayoutImpl(opts) {
    options->type("cubed-sphere");
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

  void forward(MeshBlockImpl const *pmb, Variables &vars) override;

  void serialize(MeshBlockImpl const *pmb, Variables &vars) override;

  void deserialize(MeshBlockImpl const *pmb, Variables &vars) const override;

 private:
  //! \brief Project covariant velocities to cartesian velocities
  void _covariant_to_cartesian(MeshBlockImpl const *pmb,
                               std::tuple<int, int, int> offset,
                               torch::Tensor vz, torch::Tensor vx,
                               torch::Tensor vy) const;

  //! \brief Interpolate transmitted variable to local ghost zones
  void _interpolate_to_local(MeshBlockImpl const *pmb,
                             std::tuple<int, int, int> offset,
                             torch::Tensor var) const;

  //! \brief Deproject cartesian velocities to covariant velocities
  void _cartesian_to_covariant(MeshBlockImpl const *pmb,
                               std::tuple<int, int, int> offset,
                               torch::Tensor vz, torch::Tensor vx,
                               torch::Tensor vy) const;

  //! \brief Global rank layout: face-major, Z-order within face
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
  int _face_local_rank(int rx, int ry) const {
    return _rankof[linear_index2(pxy(), pxy(), ry, rx)];
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
};
TORCH_MODULE(CubedSphereLayout);

}  // namespace snap
