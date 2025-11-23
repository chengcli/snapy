#pragma once

// C/C++
#include <memory>

// torch
#include <torch/nn/module.h>
#include <torch/nn/modules/common.h>
#include <torch/nn/modules/container/any.h>

namespace YAML {
class Node;
}

namespace snap {
class EquationOfStateImpl;
struct EquationOfStateOptions;
//! Choose equation of state
/*!
 * Supported Equation of State types
 * - ideal-gas
 * - ideal-moist
 * - moist-mixture
 * - ShallowWaterXY
 * - ShallowWaterYZ
 */
std::shared_ptr<EquationOfStateImpl> register_module_op(
    torch::nn::Module *p, std::string name, EquationOfStateOptions const &op);

class RiemannSolverImpl;
struct RiemannSolverOptions;
//! Choose Riemann solver
/*!
 * Supported Riemann Solver types
 * - lmars
 * - roe
 * - upwind
 * - hllc
 * - shallow-roe
 * - plume-roe
 */
std::shared_ptr<RiemannSolverImpl> register_module_op(
    torch::nn::Module *p, std::string name, RiemannSolverOptions const &m);

class CoordinateImpl;
struct CoordinateOptions;
//! Choose coordinate system
/*!
 * Supported Coordinate system types
 * - cartesian
 * - cylindrical
 * - spherical-polar
 * - cubed-sphere
 */
std::shared_ptr<CoordinateImpl> register_module_op(torch::nn::Module *p,
                                                   std::string name,
                                                   CoordinateOptions const &op);

class InterpImpl;
struct InterpOptions;

//! Choose reconstruction method
/*!
 * Supported reconstruction methods
 * - dc
 * - plm
 * - ppm
 * - cp3
 * - cp5
 * - weno3
 * - weno5
 */
std::shared_ptr<InterpImpl> register_module_op(torch::nn::Module *p,
                                               std::string name,
                                               InterpOptions const &op);

struct HydroOptions;
struct LayoutOptions;
//! Register forcing options
/*!
 * Supported forcing options
 * - const-gravity
 * - coriolis
 * - diffusion
 * - fric-heat
 * - body-heat
 * - bot-heat
 * - top-cool
 * - relax-bot-comp
 * - relax-bot-temp
 * - relax-bot-velo
 * - top-sponge-lyr
 * - bot-sponge-lyr
 * - plume-forcing
 */
void register_forcings_options(HydroOptions &op, YAML::Node const &config,
                               LayoutOptions const &layout);

//! Register forcing modules based on HydroOptions
std::vector<std::string> register_forcings_module(
    HydroOptions const &opts, std::vector<torch::nn::AnyModule> &forcings);
};  // namespace snap
