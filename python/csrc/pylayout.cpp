// pybind11
#include <pybind11/functional.h>
#include <pybind11/stl.h>

// fmt
#include <fmt/format.h>

// torch
#include <torch/extension.h>
#include <torch/nn/modules/container/any.h>

// snap
#include <snap/layout/distribute_env.hpp>
#include <snap/layout/layout.hpp>

// python
#include "pyoptions.hpp"

namespace py = pybind11;

void bind_layout(py::module& m) {
  auto pyLayoutOptions = py::class_<snap::LayoutOptions>(m, "LayoutOptions");

  pyLayoutOptions.def(py::init<>())
      .def("__repr__",
           [](const snap::LayoutOptions& a) {
             std::stringstream ss;
             a.report(ss);
             return fmt::format("LayoutOptions(\n{})", ss.str());
           })
      .ADD_OPTION(std::string, snap::LayoutOptions, type)
      .ADD_OPTION(int, snap::LayoutOptions, px)
      .ADD_OPTION(int, snap::LayoutOptions, py)
      .ADD_OPTION(int, snap::LayoutOptions, pz)
      .ADD_OPTION(bool, snap::LayoutOptions, periodic_x)
      .ADD_OPTION(bool, snap::LayoutOptions, periodic_y)
      .ADD_OPTION(bool, snap::LayoutOptions, periodic_z);

  auto pyDistributeEnvOptions =
      py::class_<snap::DistributeEnvOptions>(m, "DistributeEnvOptions");

  pyDistributeEnvOptions.def(py::init<>())
      .def("__repr__",
           [](const snap::DistributeEnvOptions& a) {
             std::stringstream ss;
             a.report(ss);
             return fmt::format("DistributeEnvOptions(\n{})", ss.str());
           })
      .ADD_OPTION(std::string, snap::DistributeEnvOptions, backend)
      .ADD_OPTION(std::string, snap::DistributeEnvOptions, master_addr)
      .ADD_OPTION(int, snap::DistributeEnvOptions, root_rank)
      .ADD_OPTION(int, snap::DistributeEnvOptions, rank)
      .ADD_OPTION(int, snap::DistributeEnvOptions, local_rank)
      .ADD_OPTION(int, snap::DistributeEnvOptions, world_size)
      .ADD_OPTION(int, snap::DistributeEnvOptions, master_port)
      .ADD_OPTION(bool, snap::DistributeEnvOptions, verbose);

  /*auto pySlabLayout = py::class_<snap::SlabLayout>(m, "SlabLayout");

  pySlabLayout
      .def(py::init<int, int, bool, bool>(), py::arg("px"), py::arg("py"),
           py::arg("periodic_x") = false, py::arg("periodic_y") = false)
      .def("__repr__",
           [](const snap::SlabLayout& a) {
             std::stringstream ss;
             a.report(ss);
             return fmt::format("SlabLayout(\n{})", ss.str());
           })
      .def("get_procs", &snap::SlabLayout::get_procs)
      .def("rank_of", &snap::SlabLayout::rank_of, py::arg("rx"), py::arg("ry"))
      .def("loc_of", &snap::SlabLayout::loc_of, py::arg("rank"))
      .def(
          "neighbor_rank",
          [](const snap::SlabLayout& self, int rx, int ry, int dx, int dy,
             int dz) { return self.neighbor_rank(rx, ry, dx, dy); },
          py::arg("rx"), py::arg("ry"), py::arg("dx"), py::arg("dy"),
          py::arg("dz") = 0);

  auto pyCubedLayout = py::class_<snap::CubedLayout>(m, "CubedLayout");

  pyCubedLayout
      .def(py::init<int, int, int, bool, bool, bool>(), py::arg("px"),
           py::arg("py"), py::arg("pz"), py::arg("periodic_x") = false,
           py::arg("periodic_y") = false, py::arg("periodic_z") = false)
      .def("__repr__",
           [](const snap::CubedLayout& a) {
             std::stringstream ss;
             a.report(ss);
             return fmt::format("CubedLayout(\n{})", ss.str());
           })
      .def("get_procs", &snap::CubedLayout::get_procs)
      .def("rank_of", &snap::CubedLayout::rank_of, py::arg("rx"), py::arg("ry"),
           py::arg("rz"))
      .def("loc_of", &snap::CubedLayout::loc_of, py::arg("rank"))
      .def("neighbor_rank", &snap::CubedLayout::neighbor_rank, py::arg("rx"),
           py::arg("ry"), py::arg("rz"), py::arg("dx"), py::arg("dy"),
           py::arg("dz"));

  auto pyCubedSphereLayout =
      py::class_<snap::CubedSphereLayout>(m, "CubedSphereLayout");

  pyCubedSphereLayout.def(py::init<int>(), py::arg("pxy"))
      .def("__repr__",
           [](const snap::CubedSphereLayout& a) {
             std::stringstream ss;
             a.report(ss);
             return fmt::format("CubedSphereLayout(\n{})", ss.str());
           })
      .def("get_procs", &snap::CubedSphereLayout::get_procs)
      .def("rank_of", &snap::CubedSphereLayout::rank_of, py::arg("face"),
           py::arg("rx"), py::arg("ry"))
      .def("loc_of", &snap::CubedSphereLayout::loc_of, py::arg("rank"))
      .def(
          "neighbor_rank",
          [](const snap::CubedSphereLayout& self, int face, int rx, int ry,
             int dx, int dy,
             int dz) { return self.neighbor_rank(face, rx, ry, dx, dy); },
          py::arg("face"), py::arg("rx"), py::arg("ry"), py::arg("dx"),
          py::arg("dy"), py::arg("dz") = 0);
  */

  // distribution functions
  m.def("get_buffer_id", &snap::get_buffer_id)
      .def("get_rank_from_env", &snap::get_rank)
      .def("get_local_rank", &snap::get_local_rank);
}
