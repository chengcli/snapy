// torch
#include <torch/extension.h>

// fvm
#include <fvm/coord/coord_formatter.hpp>
#include <fvm/coord/coordinate.hpp>

// python
#include "pyoptions.hpp"

namespace py = pybind11;

void bind_coord(py::module &m) {
  py::class_<canoe::CoordinateOptions>(m, "CoordinateOptions")
      .def(py::init<>())
      .def(py::init<canoe::ParameterInput>())
      .def("__repr__",
           [](const canoe::CoordinateOptions &a) {
             return fmt::format("CoordinateOptions{}", a);
           })
      .ADD_OPTION(double, canoe::CoordinateOptions, x1min)
      .ADD_OPTION(double, canoe::CoordinateOptions, x1max)
      .ADD_OPTION(double, canoe::CoordinateOptions, x2min)
      .ADD_OPTION(double, canoe::CoordinateOptions, x2max)
      .ADD_OPTION(double, canoe::CoordinateOptions, x3min)
      .ADD_OPTION(double, canoe::CoordinateOptions, x3max)
      .ADD_OPTION(int, canoe::CoordinateOptions, nx1)
      .ADD_OPTION(int, canoe::CoordinateOptions, nx2)
      .ADD_OPTION(int, canoe::CoordinateOptions, nx3);

  ADD_CANOE_MODULE(Cartesian, CoordinateOptions);
}
