// torch
#include <torch/extension.h>

// snap
#include <snap/coord/coord_utils.hpp>
#include <snap/coord/coordinate.hpp>
#include <snap/coord/cubed_sphere_utils.hpp>
#include <snap/coord/gnomonic_equiangle.hpp>
#include <snap/layout/cubed_sphere_layout.hpp>

// python
#include "pyoptions.hpp"

namespace py = pybind11;

void bind_coord(py::module &m) {
  auto pyCoordinateOptions =
      py::class_<snap::CoordinateOptionsImpl, snap::CoordinateOptions>(
          m, "CoordinateOptions");

  pyCoordinateOptions.def(py::init<>())
      .def("__repr__",
           [](const snap::CoordinateOptions &a) {
             std::stringstream ss;
             a->report(ss);
             return fmt::format("CoordinateOptions(\n{})", ss.str());
           })
      .ADD_OPTION(double, snap::CoordinateOptionsImpl, x1min)
      .ADD_OPTION(double, snap::CoordinateOptionsImpl, x1max)
      .ADD_OPTION(double, snap::CoordinateOptionsImpl, x2min)
      .ADD_OPTION(double, snap::CoordinateOptionsImpl, x2max)
      .ADD_OPTION(double, snap::CoordinateOptionsImpl, x3min)
      .ADD_OPTION(double, snap::CoordinateOptionsImpl, x3max)
      .ADD_OPTION(int, snap::CoordinateOptionsImpl, nx1)
      .ADD_OPTION(int, snap::CoordinateOptionsImpl, nx2)
      .ADD_OPTION(int, snap::CoordinateOptionsImpl, nx3)
      .ADD_OPTION(int, snap::CoordinateOptionsImpl, nghost);

  auto pyCoordinate =
      py::class_<snap::CoordinateImpl, snap::Coordinate>(m, "Coordinate");

  pyCoordinate.def(py::init<snap::CoordinateOptions>(), py::arg("options"))
      .def("__repr__",
           [](const snap::CoordinateImpl &self) {
             std::stringstream ss;
             self.options->report(ss);
             return fmt::format("Coordinate(\n{})", ss.str());
           })
      .def("ifirst", [](snap::CoordinateImpl &self) { return self.is(); })
      .def("ilast", [](snap::CoordinateImpl &self) { return self.ie() + 1; })
      .def("jfirst", [](snap::CoordinateImpl &self) { return self.js(); })
      .def("jlast", [](snap::CoordinateImpl &self) { return self.je() + 1; })
      .def("kfirst", [](snap::CoordinateImpl &self) { return self.ks(); })
      .def("klast", [](snap::CoordinateImpl &self) { return self.ke() + 1; })
      .def("center_width1",
           [](snap::CoordinateImpl &self) { return self.center_width1(); })
      .def("center_width2",
           [](snap::CoordinateImpl &self) { return self.center_width2(); })
      .def("center_width3",
           [](snap::CoordinateImpl &self) { return self.center_width3(); })
      .def("face_area1",
           [](snap::CoordinateImpl &self) { return self.face_area1(); })
      .def("face_area2",
           [](snap::CoordinateImpl &self) { return self.face_area2(); })
      .def("face_area3",
           [](snap::CoordinateImpl &self) { return self.face_area3(); })
      .def("cell_volume",
           [](snap::CoordinateImpl &self) { return self.cell_volume(); });

  auto pyCartesian =
      py::class_<snap::CartesianImpl, snap::CoordinateImpl,
                 std::shared_ptr<snap::CartesianImpl>>(m, "Cartesian");

  pyCartesian.def("ifirst", [](snap::CartesianImpl &self) { return self.is(); })
      .def("ilast", [](snap::CartesianImpl &self) { return self.ie() + 1; });

  torch::python::add_module_bindings(pyCartesian)
      .def(py::init<snap::CoordinateOptions, torch::nn::Module *>(),
           py::arg("options"), py::arg("hydro") = nullptr)
      .def("buffer",
           [](snap::CartesianImpl &self, std::string name) {
             return self.named_buffers()[name];
           })
      .def("module", [](snap::CartesianImpl &self, std::string name) {
        return self.named_modules()[name];
      });

  auto pyGnomonicEquiangle =
      py::class_<snap::GnomonicEquiangleImpl, snap::CoordinateImpl,
                 std::shared_ptr<snap::GnomonicEquiangleImpl>>(
          m, "GnomonicEquiangle");

  pyGnomonicEquiangle
      .def("ifirst",
           [](snap::GnomonicEquiangleImpl &self) { return self.is(); })
      .def("ilast",
           [](snap::GnomonicEquiangleImpl &self) { return self.ie() + 1; });

  torch::python::add_module_bindings(pyGnomonicEquiangle)
      .def(py::init<snap::CoordinateOptions, torch::nn::Module *>(),
           py::arg("options"), py::arg("hydro") = nullptr)
      .def("buffer",
           [](snap::GnomonicEquiangleImpl &self, std::string name) {
             return self.named_buffers()[name];
           })
      .def("module", [](snap::GnomonicEquiangleImpl &self, std::string name) {
        return self.named_modules()[name];
      });

  auto m_coord = m.def_submodule("coord", "Coordinate submodule");
  m_coord.def("coord_vec_lower_", &snap::coord_vec_lower_)
      .def("coord_vec_raise_", &snap::coord_vec_raise_)
      .def("cs_cart_to_contra_", &snap::cs_cart_to_contra_)
      .def("cs_contra_to_cart_", &snap::cs_contra_to_cart_)
      .def("cs_ab_to_lonlat", &snap::cs_ab_to_lonlat)
      .def("get_cs_face_name", [](int face_id) {
        return std::string(snap::CS_FACE_NAMES[face_id]);
      });
}
