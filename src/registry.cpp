// snap
#include <snap/coord/coordinate.hpp>
#include <snap/eos/aneos.hpp>
#include <snap/eos/ideal_gas.hpp>
#include <snap/eos/ideal_moist.hpp>
#include <snap/eos/moist_mixture.hpp>
#include <snap/eos/shallow_water.hpp>
#include <snap/recon/interpolation.hpp>
#include <snap/riemann/riemann_solver.hpp>

namespace snap {
EquationOfState register_module_op(torch::nn::Module *p, std::string name,
                                   EquationOfStateOptions const &op) {
  if (op.type() == "moist-mixture") {
    return p->register_module(name, MoistMixture(op));
  } else if (op.type() == "ideal-gas") {
    return p->register_module(name, IdealGas(op));
  } else if (op.type() == "ideal-moist") {
    return p->register_module(name, IdealMoist(op));
  } else if (op.type() == "aneos") {
    return p->register_module(name, ANEOS(op));
  } else if (op.type() == "shallow-water") {
    return p->register_module(name, ShallowWater(op));
  } else {
    throw std::runtime_error("register_module: unknown type " + op.type());
  }
}

RiemannSolver register_module_op(torch::nn::Module *p, std::string name,
                                 RiemannSolverOptions const &op) {
  if (op.type() == "roe") {
    return p->register_module(name, RoeSolver(op));
  } else if (op.type() == "lmars") {
    return p->register_module(name, LmarsSolver(op));
  } else if (op.type() == "hllc") {
    return p->register_module(name, HLLCSolver(op));
  } else if (op.type() == "upwind") {
    return p->register_module(name, UpwindSolver(op));
  } else if (op.type() == "shallow-roe") {
    return p->register_module(name, ShallowRoeSolver(op));
  } else {
    throw std::runtime_error("register_module: unknown type " + op.type());
  }
}

Coordinate register_module_op(torch::nn::Module *p, std::string name,
                              CoordinateOptions const &op) {
  if (op.type() == "cartesian") {
    return p->register_module(name, Cartesian(op));
  } else if (op.type() == "cylindrical") {
    return p->register_module(name, Cylindrical(op));
  } else if (op.type() == "spherical-polar") {
    return p->register_module(name, SphericalPolar(op));
  } else if (op.type() == "cubed-sphere") {
    return p->register_module(name, GnomonicEquiangle(op));
  } else {
    throw std::runtime_error("register_module: unknown type " + op.type());
  }
}

Interp register_module_op(torch::nn::Module *p, std::string name,
                          InterpOptions const &op) {
  if (op.type() == "dc") {
    return p->register_module(name, DonorCellInterp(op));
  } else if (op.type() == "plm") {
    return p->register_module(name, PLMInterp(op));
  } else if (op.type() == "ppm") {
    return p->register_module(name, PPMInterp(op));
  } else if (op.type() == "cp3") {
    return p->register_module(name, Center3Interp(op));
  } else if (op.type() == "cp5") {
    return p->register_module(name, Center5Interp(op));
  } else if (op.type() == "weno3") {
    if (name.back() == '1') {
      return p->register_module(name, Weno3Interp(op));
    } else if (name.back() == '2') {
      return p->register_module(name, Center3Interp(op));
    } else {
      throw std::runtime_error("register_module: unknown name " + name);
    }
  } else if (op.type() == "weno5") {
    if (name.back() == '1') {
      return p->register_module(name, Weno5Interp(op));
    } else if (name.back() == '2') {
      return p->register_module(name, Center5Interp(op));
    } else {
      throw std::runtime_error("register_module: unknown name " + name);
    }
  } else {
    throw std::runtime_error("register_module: unknown type " + op.type());
  }
}

}  // namespace snap
