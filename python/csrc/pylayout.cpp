// pybind11
#include <pybind11/functional.h>

// fmt
#include <fmt/format.h>

// torch
#include <torch/extension.h>
#include <torch/nn/modules/container/any.h>

// canoe
#include <snap/layout/cubed_layout.hpp>
#include <snap/layout/cubed_sphere_layout.hpp>
#include <snap/layout/slab_layout.hpp>

// python
#include "pyoptions.hpp"

namespace py = pybind11;

void bind_layout(py::module &m) {
  auto pySlabLayout = py::class_<canoe::SlabLayout>(m, "SlabLayout");

  pySlabLayout
      .def(py::init<int, int, bool, bool>(), py::arg("px"), py::arg("py"),
           py::arg("periodic_x") = false, py::arg("periodic_y") = false)
      .def("__repr__",
           [](const canoe::SlabLayout &a) {
             std::stringstream ss;
             a.report(ss);
             return fmt::format("SlabLayout(\n{})", ss.str());
           })
      .def("get_procs", &canoe::SlabLayout::get_procs)
      .def("rank_of", &canoe::SlabLayout::rank_of, py::arg("rx"), py::arg("ry"))
      .def("loc_of", &canoe::SlabLayout::loc_of, py::arg("rank"))
      .def(
          "neighbor_rank",
          [](const canoe::SlabLayout &self, int rx, int ry, int dx, int dy,
             int dz) { return self.neighbor_rank(rx, ry, dx, dy); },
          py::arg("rx"), py::arg("ry"), py::arg("dx"), py::arg("dy"),
          py::arg("dz") = 0);

  auto pyCubedLayout = py::class_<canoe::CubedLayout>(m, "CubedLayout");

  pyCubedLayout
      .def(py::init<int, int, int, bool, bool, bool>(), py::arg("px"),
           py::arg("py"), py::arg("pz"), py::arg("periodic_x") = false,
           py::arg("periodic_y") = false, py::arg("periodic_z") = false)
      .def("__repr__",
           [](const canoe::CubedLayout &a) {
             std::stringstream ss;
             a.report(ss);
             return fmt::format("CubedLayout(\n{})", ss.str());
           })
      .def("get_procs", &canoe::CubedLayout::get_procs)
      .def("rank_of", &canoe::CubedLayout::rank_of, py::arg("rx"),
           py::arg("ry"), py::arg("rz"))
      .def("loc_of", &canoe::CubedLayout::loc_of, py::arg("rank"))
      .def("neighbor_rank", &canoe::CubedLayout::neighbor_rank, py::arg("rx"),
           py::arg("ry"), py::arg("rz"), py::arg("dx"), py::arg("dy"),
           py::arg("dz"));

  auto pyCubedSphereLayout =
      py::class_<canoe::CubedSphereLayout>(m, "CubedSphereLayout");

  pyCubedSphereLayout.def(py::init<int>(), py::arg("pxy"))
      .def("__repr__",
           [](const canoe::CubedSphereLayout &a) {
             std::stringstream ss;
             a.report(ss);
             return fmt::format("CubedSphereLayout(\n{})", ss.str());
           })
      .def("get_procs", &canoe::CubedSphereLayout::get_procs)
      .def("rank_of", &canoe::CubedSphereLayout::rank_of, py::arg("face"),
           py::arg("rx"), py::arg("ry"))
      .def("loc_of", &canoe::CubedSphereLayout::loc_of, py::arg("rank"))
      .def(
          "neighbor_rank",
          [](const canoe::CubedSphereLayout &self, int face, int rx, int ry,
             int dx, int dy,
             int dz) { return self.neighbor_rank(face, rx, ry, dx, dy); },
          py::arg("face"), py::arg("rx"), py::arg("ry"), py::arg("dx"),
          py::arg("dy"), py::arg("dz") = 0);
}
