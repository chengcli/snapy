#ifndef SRC_SNAP_TURBULENCE_TURBULENCE_MODEL_HPP_
#define SRC_SNAP_TURBULENCE_TURBULENCE_MODEL_HPP_

// C/C++
#include <memory>

// Athena++ headers
#include <athena/athena.hpp>
#include <athena/athena_arrays.hpp>
#include <athena/bvals/cc/bvals_cc.hpp>

class MeshBlock;
class ParameterInput;

//! \brief Base class for turbulence models
//!
//! Provides interface for different turbulence modeling approaches
//! in atmospheric dynamics simulations.
class TurbulenceModel {
 public:
  //! \brief Constructor
  //! \param[in] pmb Pointer to mesh block
  //! \param[in] pin Pointer to parameter input
  TurbulenceModel(MeshBlock *pmb, ParameterInput *pin);

  //! \brief Virtual destructor
  virtual ~TurbulenceModel();

  // access members
  AthenaArray<Real> w, u;  //!< Primitive and conserved variables

  // public data
  AthenaArray<Real> mut;  //!< Dynamic turbulent viscosity

  // public functions:
  //! \brief Drive turbulence evolution
  //! \param[in] dt Time step
  virtual void DriveTurbulence(Real dt) {}

  //! \brief Initialize turbulence model
  virtual void Initialize() {}

  //! \brief Set diffusivity coefficients
  //! \param[out] nu Viscosity array
  //! \param[out] kappa Thermal diffusivity array
  //! \param[in] w Primitive variables
  //! \param[in] bc Boundary conditions
  //! \param[in] il Lower i-index
  //! \param[in] iu Upper i-index
  //! \param[in] jl Lower j-index
  //! \param[in] ju Upper j-index
  //! \param[in] kl Lower k-index
  //! \param[in] ku Upper k-index
  virtual void SetDiffusivity(AthenaArray<Real> &nu, AthenaArray<Real> &kappa,
                              const AthenaArray<Real> &w,
                              const AthenaArray<Real> &bc, int il, int iu,
                              int jl, int ju, int kl, int ku) {}

 protected:
  MeshBlock *pmy_block;  //!< Pointer to parent mesh block
};

//! \brief K-Epsilon turbulence model implementation
//!
//! Two-equation turbulence model that solves transport equations for
//! turbulent kinetic energy (k) and its dissipation rate (epsilon).
class KEpsilonTurbulence : public TurbulenceModel {
 public:
  //! \brief Constructor
  //! \param[in] pmb Pointer to mesh block
  //! \param[in] pin Pointer to parameter input
  KEpsilonTurbulence(MeshBlock *pmb, ParameterInput *pin);

  //! \brief Destructor
  ~KEpsilonTurbulence() {}

  //! \brief Drive turbulence evolution using k-epsilon model
  //! \param[in] dt Time step
  void DriveTurbulence(Real dt);

  //! \brief Initialize k-epsilon turbulence model
  void Initialize();

  //! \brief Set diffusivity using k-epsilon model
  //! \param[out] nu Viscosity array
  //! \param[out] kappa Thermal diffusivity array
  //! \param[in] w Primitive variables
  //! \param[in] bc Boundary conditions
  //! \param[in] il Lower i-index
  //! \param[in] iu Upper i-index
  //! \param[in] jl Lower j-index
  //! \param[in] ju Upper j-index
  //! \param[in] kl Lower k-index
  //! \param[in] ku Upper k-index
  void SetDiffusivity(AthenaArray<Real> &nu, AthenaArray<Real> &kappa,
                      const AthenaArray<Real> &w, const AthenaArray<Real> &bc,
                      int il, int iu, int jl, int ju, int kl, int ku);

 private:
  Real cmu_, c1_, c2_, sigk_, sige_;  //!< K-epsilon model constants
};

using TurbulenceModelPtr =
    std::shared_ptr<TurbulenceModel>;  //!< Shared pointer to turbulence model

//! \brief Factory class for creating turbulence models
class TurbulenceFactory {
 public:
  //! \brief Create a turbulence model
  //! \param[in] pmb Pointer to mesh block
  //! \param[in] pin Pointer to parameter input
  //! \return Shared pointer to created turbulence model
  static TurbulenceModelPtr Create(MeshBlock *pmb, ParameterInput *pin);
};

#endif  // SRC_SNAP_TURBULENCE_TURBULENCE_MODEL_HPP_
