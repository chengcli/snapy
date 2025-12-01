// torch
#include <torch/extension.h>

// snap
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
      .ADD_OPTION(int, snap::CoordinateOptions, nx3)
      .ADD_OPTION(int, snap::CoordinateOptions, nghost);

  // temporary solution

  ADD_SNAP_MODULE(Cartesian, CoordinateOptions)
      .def("__repr__",
           [](const snap::CartesianImpl &self) {
             std::stringstream ss;
             self.print(ss);
             return fmt::format("Coordinate(\n{})", ss.str());
           })
      .def("ifirst", [](snap::CartesianImpl &self) { return self.is(); })
      .def("ilast", [](snap::CartesianImpl &self) { return self.ie() + 1; })
      .def("jfirst", [](snap::CartesianImpl &self) { return self.js(); })
      .def("jlast", [](snap::CartesianImpl &self) { return self.je() + 1; })
      .def("kfirst", [](snap::CartesianImpl &self) { return self.ks(); })
      .def("klast", [](snap::CartesianImpl &self) { return self.ke() + 1; })
      .def("center_width1",
           [](snap::CartesianImpl &self) { return self.center_width1(); })
      .def("center_width2",
           [](snap::CartesianImpl &self) { return self.center_width2(); })
      .def("center_width3",
           [](snap::CartesianImpl &self) { return self.center_width3(); })
      .def("face_area1",
           [](snap::CartesianImpl &self) { return self.face_area1(); })
      .def("face_area2",
           [](snap::CartesianImpl &self) { return self.face_area2(); })
      .def("face_area3",
           [](snap::CartesianImpl &self) { return self.face_area3(); })
      .def("cell_volume",
           [](snap::CartesianImpl &self) { return self.cell_volume(); });
}
