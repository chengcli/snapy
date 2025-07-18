// torch
#include <torch/extension.h>

// snap
#include <snap/coord/coord_formatter.hpp>
#include <snap/coord/coordinate.hpp>

// python
#include "pyoptions.hpp"

namespace py = pybind11;

void bind_coord(py::module &m) {
  auto pyCoordinateOptions =
      py::class_<snap::CoordinateOptions>(m, "CoordinateOptions");

  pyCoordinateOptions.def(py::init<>())
      .def("__repr__",
           [](const snap::CoordinateOptions &a) {
             std::stringstream ss;
             a.report(ss);
             return fmt::format("CoordinateOptions(\n{})", ss.str());
           })
      .ADD_OPTION(double, snap::CoordinateOptions, x1min)
      .ADD_OPTION(double, snap::CoordinateOptions, x1max)
      .ADD_OPTION(double, snap::CoordinateOptions, x2min)
      .ADD_OPTION(double, snap::CoordinateOptions, x2max)
      .ADD_OPTION(double, snap::CoordinateOptions, x3min)
      .ADD_OPTION(double, snap::CoordinateOptions, x3max)
      .ADD_OPTION(int, snap::CoordinateOptions, nx1)
      .ADD_OPTION(int, snap::CoordinateOptions, nx2)
      .ADD_OPTION(int, snap::CoordinateOptions, nx3);

  ADD_SNAP_MODULE(Cartesian, CoordinateOptions);
}
