// pybind11
#include <pybind11/functional.h>
#include <pybind11/stl.h>

// fmt
#include <fmt/format.h>

// torch
#include <torch/extension.h>
#include <torch/nn/modules/container/any.h>

// snap
#include <snap/layout/cubed_sphere_layout.hpp>
#include <snap/layout/layout.hpp>
#include <snap/mesh/meshblock.hpp>

// python
#include "pyoptions.hpp"

namespace py = pybind11;

void bind_layout(py::module &m) {
  auto pyLayoutOptions =
      py::class_<snap::LayoutOptionsImpl, snap::LayoutOptions>(m,
                                                               "LayoutOptions");

  pyLayoutOptions.def(py::init<>())
      .def("__repr__",
           [](const snap::LayoutOptions &a) {
             std::stringstream ss;
             a->report(ss);
             return fmt::format("LayoutOptions(\n{})", ss.str());
           })
      .ADD_OPTION(std::string, snap::LayoutOptionsImpl, type)
      .ADD_OPTION(int, snap::LayoutOptionsImpl, px)
      .ADD_OPTION(int, snap::LayoutOptionsImpl, py)
      .ADD_OPTION(int, snap::LayoutOptionsImpl, pz)
      .ADD_OPTION(bool, snap::LayoutOptionsImpl, periodic_x)
      .ADD_OPTION(bool, snap::LayoutOptionsImpl, periodic_y)
      .ADD_OPTION(bool, snap::LayoutOptionsImpl, periodic_z)
      .ADD_OPTION(bool, snap::LayoutOptionsImpl, verbose)
      .ADD_OPTION(bool, snap::LayoutOptionsImpl, no_backend)
      .ADD_OPTION(std::string, snap::LayoutOptionsImpl, backend)
      .ADD_OPTION(std::string, snap::LayoutOptionsImpl, master_addr)
      .ADD_OPTION(int, snap::LayoutOptionsImpl, master_port)
      .ADD_OPTION(int, snap::LayoutOptionsImpl, root_rank)
      .ADD_OPTION(int, snap::LayoutOptionsImpl, world_size)
      .ADD_OPTION(int, snap::LayoutOptionsImpl, rank)
      .ADD_OPTION(int, snap::LayoutOptionsImpl, local_rank);

  py::class_<snap::LayoutImpl, snap::Layout>(m, "Layout")
      .def(py::init<snap::LayoutOptions>(), py::arg("options"))
      .def("__repr__",
           [](const snap::LayoutImpl &self) {
             std::stringstream ss;
             self.options->report(ss);
             return fmt::format("Layout(\n{})", ss.str());
           })
      .def("get_procs", &snap::LayoutImpl::get_procs)
      .def("rank_of", &snap::LayoutImpl::rank_of)
      .def("loc_of", &snap::LayoutImpl::loc_of)
      .def("neighbor_rank", &snap::LayoutImpl::neighbor_rank);

  auto pySlabLayout =
      py::class_<snap::SlabLayoutImpl, snap::LayoutImpl,
                 std::shared_ptr<snap::SlabLayoutImpl>>(m, "SlabLayout");

  pySlabLayout.def(py::init<snap::LayoutOptions>(), py::arg("options"))
      .def("__repr__", [](const snap::SlabLayoutImpl &self) {
        std::stringstream ss;
        self.pretty_print(ss);
        return fmt::format("SlabLayout(\n{})", ss.str());
      });

  torch::python::add_module_bindings(pySlabLayout)
      .def("buffer",
           [](snap::SlabLayoutImpl &self, std::string name) {
             return self.named_buffers()[name];
           })
      .def("module", [](snap::SlabLayoutImpl &self, std::string name) {
        return self.named_modules()[name];
      });

  auto pyCubedLayout =
      py::class_<snap::CubedLayoutImpl, snap::LayoutImpl,
                 std::shared_ptr<snap::CubedLayoutImpl>>(m, "CubedLayout");

  pyCubedLayout.def(py::init<snap::LayoutOptions>(), py::arg("options"))
      .def("__repr__", [](const snap::CubedSphereLayoutImpl &self) {
        std::stringstream ss;
        self.pretty_print(ss);
        return fmt::format("CubedSphereLayout(\n{})", ss.str());
      });

  torch::python::add_module_bindings(pyCubedLayout)
      .def("buffer",
           [](snap::CubedLayoutImpl &self, std::string name) {
             return self.named_buffers()[name];
           })
      .def("module", [](snap::CubedLayoutImpl &self, std::string name) {
        return self.named_modules()[name];
      });

  auto pyCubedSphereLayout =
      py::class_<snap::CubedSphereLayoutImpl, snap::LayoutImpl,
                 std::shared_ptr<snap::CubedSphereLayoutImpl>>(
          m, "CubedSphereLayout");

  pyCubedSphereLayout.def(py::init<>())
      .def(py::init<snap::LayoutOptions>(), py::arg("options"))
      .def("__repr__", [](const snap::CubedSphereLayoutImpl &self) {
        std::stringstream ss;
        self.pretty_print(ss);
        return fmt::format("CubedSphereLayout(\n{})", ss.str());
      });

  torch::python::add_module_bindings(pyCubedSphereLayout)
      .def("buffer",
           [](snap::CubedSphereLayoutImpl &self, std::string name) {
             return self.named_buffers()[name];
           })
      .def("module", [](snap::CubedSphereLayoutImpl &self, std::string name) {
        return self.named_modules()[name];
      });

  // distribution functions
  auto m_dist = m.def_submodule("distributed", "Distributed module");
  m_dist.def("get_rank", &snap::get_rank)
      .def("get_local_rank", &snap::get_local_rank)
      .def("get_layout", &snap::MeshBlockImpl::get_layout);
}
