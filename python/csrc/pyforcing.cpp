// torch
#include <torch/extension.h>
#include <torch/nn/modules/container/any.h>

// snap
#include <snap/forcing/forcing.hpp>
#include <snap/forcing/forcing_formatter.hpp>

// python
#include "pyoptions.hpp"

namespace py = pybind11;

void bind_forcing(py::module &m) {
  py::class_<snap::ConstGravityOptions>(m, "ConstGravityOptions")
      .def(py::init<>())
      .def("__repr__",
           [](const snap::ConstGravityOptions &a) {
             return fmt::format("ConstGravityOptions{}", a);
           })
      .ADD_OPTION(double, snap::ConstGravityOptions, grav1)
      .ADD_OPTION(double, snap::ConstGravityOptions, grav2)
      .ADD_OPTION(double, snap::ConstGravityOptions, grav3);

  py::class_<snap::CoriolisOptions>(m, "CoriolisOptions")
      .def(py::init<>())
      .def("__repr__",
           [](const snap::CoriolisOptions &a) {
             return fmt::format("CoriolisOptions{}", a);
           })
      .ADD_OPTION(double, snap::CoriolisOptions, omega1)
      .ADD_OPTION(double, snap::CoriolisOptions, omega2)
      .ADD_OPTION(double, snap::CoriolisOptions, omega3)
      .ADD_OPTION(double, snap::CoriolisOptions, omegax)
      .ADD_OPTION(double, snap::CoriolisOptions, omegay)
      .ADD_OPTION(double, snap::CoriolisOptions, omegaz);
}
