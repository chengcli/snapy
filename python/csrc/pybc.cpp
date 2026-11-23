// pybind11
#include <pybind11/functional.h>

// torch
#include <torch/extension.h>
#include <torch/nn/modules/container/any.h>

// snap
#include <snap/bc/bc.hpp>
#include <snap/coord/coordinate.hpp>
#include <snap/eb/geometry.hpp>

// python
#include "pyoptions.hpp"

namespace py = pybind11;

void bind_bc(py::module &m) {
  m.attr("kUnknown") = (int)snap::BoundaryFace::kUnknown;
  m.attr("kInnerX1") = (int)snap::BoundaryFace::kInnerX1;
  m.attr("kOuterX1") = (int)snap::BoundaryFace::kOuterX1;
  m.attr("kInnerX2") = (int)snap::BoundaryFace::kInnerX2;
  m.attr("kOuterX2") = (int)snap::BoundaryFace::kOuterX2;
  m.attr("kInnerX3") = (int)snap::BoundaryFace::kInnerX3;
  m.attr("kOuterX3") = (int)snap::BoundaryFace::kOuterX3;

  auto pyBoundaryFunctionOptions =
      py::class_<snap::BoundaryFuncOptions>(m, "BoundaryFuncOptions");

  pyBoundaryFunctionOptions.def(py::init<>())
      .def("__repr__",
           [](const snap::BoundaryFuncOptions &a) {
             std::stringstream ss;
             a.report(ss);
             return fmt::format("BoundaryFuncOptions(\n{})", ss.str());
           })
      .ADD_OPTION(int, snap::BoundaryFuncOptions, type)
      .ADD_OPTION(int, snap::BoundaryFuncOptions, nghost);

  py::class_<snap::EBGeometry>(m, "EmbeddedBoundaryGeometry")
      .def_static("from_level_set", &snap::EBGeometry::from_level_set,
                  py::arg("phi"), py::arg("x1f"), py::arg("x2f"),
                  py::arg("x3f"))
      .def_static("from_metrics", &snap::EBGeometry::from_metrics,
                  py::arg("volume_fraction"), py::arg("cell_centroid"),
                  py::arg("face_fraction"), py::arg("face_centroid"),
                  py::arg("wall_area_vector"), py::arg("wall_centroid"),
                  py::arg("x1f"), py::arg("x2f"), py::arg("x3f"))
      .def_readonly("volume_fraction", &snap::EBGeometry::volume_fraction)
      .def_readonly("cell_centroid", &snap::EBGeometry::cell_centroid)
      .def_readonly("face_fraction", &snap::EBGeometry::face_fraction)
      .def_readonly("face_centroid", &snap::EBGeometry::face_centroid)
      .def_readonly("wall_area_vector", &snap::EBGeometry::wall_area_vector)
      .def_readonly("wall_centroid", &snap::EBGeometry::wall_centroid)
      .def_readonly("flags", &snap::EBGeometry::flags);
}
