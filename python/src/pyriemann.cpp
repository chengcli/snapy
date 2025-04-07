// torch
#include <torch/extension.h>

// fvm
#include <fvm/riemann/riemann_formatter.hpp>
#include <fvm/riemann/riemann_solver.hpp>

// python
#include "pyoptions.hpp"

namespace py = pybind11;

void bind_riemann(py::module &m) {
  py::class_<canoe::RiemannSolverOptions>(m, "RiemannSolverOptions")
      .def(py::init<>())
      .def(py::init<canoe::ParameterInput>())
      .def("__repr__",
           [](const canoe::RiemannSolverOptions &a) {
             return fmt::format("RiemannSolverOptions{}", a);
           })
      .ADD_OPTION(std::string, canoe::RiemannSolverOptions, type);

  ADD_CANOE_MODULE(UpwindSolver, RiemannSolverOptions);
  ADD_CANOE_MODULE(RoeSolver, RiemannSolverOptions);
  ADD_CANOE_MODULE(LmarsSolver, RiemannSolverOptions);
  ADD_CANOE_MODULE(ShallowRoeSolver, RiemannSolverOptions);
}
