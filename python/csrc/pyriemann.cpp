// torch
#include <torch/extension.h>

// snap
#include <snap/riemann/riemann_formatter.hpp>
#include <snap/riemann/riemann_solver.hpp>

// python
#include "pyoptions.hpp"

namespace py = pybind11;

void bind_riemann(py::module &m) {
  auto pyRiemannSolverOptions =
      py::class_<snap::RiemannSolverOptions>(m, "RiemannSolverOptions");

  pyRiemannSolverOptions.def(py::init<>())
      .def("__repr__",
           [](const snap::RiemannSolverOptions &a) {
             std::stringstream ss;
             a.report(ss);
             return fmt::format("RiemannSolverOptions(\n{})", ss.str());
           })
      .ADD_OPTION(std::string, snap::RiemannSolverOptions, type);

  ADD_SNAP_MODULE(UpwindSolver, RiemannSolverOptions);
  ADD_SNAP_MODULE(RoeSolver, RiemannSolverOptions);
  ADD_SNAP_MODULE(LmarsSolver, RiemannSolverOptions);
  ADD_SNAP_MODULE(ShallowRoeSolver, RiemannSolverOptions);
}
