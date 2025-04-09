#pragma once

// C/C++
#include <memory>

// torch
#include <torch/nn/module.h>

namespace snap {
//! Choose between [IdealGas, IdealMoist, ShallowWaterXY, ShallowWaterYZ]
class EquationOfStateImpl;
struct EquationOfStateOptions;
std::shared_ptr<EquationOfStateImpl> register_module_op(
    torch::nn::Module *p, std::string name, EquationOfStateOptions const &op);

//! Choose between [LmarsSolver, RoeSolver, Upwind, HLLCSolver,
//! ShallowRoeSolver]
class RiemannSolverImpl;
struct RiemannSolverOptions;
std::shared_ptr<RiemannSolverImpl> register_module_op(
    torch::nn::Module *p, std::string name, RiemannSolverOptions const &m);

//! Choose between [Cartesian, Cylindrical, SphericalPolar, CubedSphere]
class CoordinateImpl;
struct CoordinateOptions;
std::shared_ptr<CoordinateImpl> register_module_op(torch::nn::Module *p,
                                                   std::string name,
                                                   CoordinateOptions const &op);

//! Choose between [DonorCellInterp, PLMInterp, Weno3Interp, Weno5Interp]
class InterpImpl;
struct InterpOptions;
std::shared_ptr<InterpImpl> register_module_op(torch::nn::Module *p,
                                               std::string name,
                                               InterpOptions const &op);
};  // namespace snap
