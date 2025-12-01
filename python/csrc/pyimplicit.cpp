// C/C++
#include <sstream>

// torch
#include <torch/extension.h>
#include <torch/nn/modules/container/any.h>

// snap
#include <snap/implicit/implicit.hpp>

// python
#include "pyoptions.hpp"

namespace py = pybind11;

void bind_implicit(py::module &m) {
  auto pyImplicitOptions =
      py::class_<snap::ImplicitOptions>(m, "ImplicitOptions");

  pyImplicitOptions.def(py::init<>())
      .def("__repr__",
           [](const snap::ImplicitOptions &a) {
             std::stringstream ss;
             a.report(ss);
             return fmt::format("ImplicitOptions(\n{})", ss.str());
           })
      .ADD_OPTION(std::string, snap::ImplicitOptions, type)
      .ADD_OPTION(double, snap::ImplicitOptions, grav)
      .ADD_OPTION(int, snap::ImplicitOptions, scheme)
      .ADD_OPTION(snap::CoordinateOptions, snap::ImplicitOptions, coord);

  ADD_SNAP_MODULE(ImplicitHydro, ImplicitOptions);
  ADD_SNAP_MODULE(ImplicitCorrection, ImplicitOptions);
}
