// torch
#include <torch/extension.h>
#include <torch/nn/modules/container/any.h>

// snap
#include <snap/forcing/forcing.hpp>

// python
#include "pyoptions.hpp"

namespace py = pybind11;

void bind_forcing(py::module& m) {
  auto pyConstGravityOptions =
      py::class_<snap::ConstGravityOptionsImpl, snap::ConstGravityOptions>(
          m, "ConstGravityOptions");

  pyConstGravityOptions.def(py::init<>())
      .def("__repr__",
           [](const snap::ConstGravityOptions& a) {
             std::stringstream ss;
             a->report(ss);
             return fmt::format("ConstGravityOptions(\n{})", ss.str());
           })
      .ADD_OPTION(double, snap::ConstGravityOptionsImpl, grav1)
      .ADD_OPTION(double, snap::ConstGravityOptionsImpl, grav2)
      .ADD_OPTION(double, snap::ConstGravityOptionsImpl, grav3);

  auto pyCoriolisOptions =
      py::class_<snap::CoriolisOptionsImpl, snap::CoriolisOptions>(
          m, "CoriolisOptions");

  pyCoriolisOptions.def(py::init<>())
      .def("__repr__",
           [](const snap::CoriolisOptions& a) {
             std::stringstream ss;
             a->report(ss);
             return fmt::format("CoriolisOptions(\n{})", ss.str());
           })
      .ADD_OPTION(double, snap::CoriolisOptionsImpl, omega1)
      .ADD_OPTION(double, snap::CoriolisOptionsImpl, omega2)
      .ADD_OPTION(double, snap::CoriolisOptionsImpl, omega3);

  auto pyDiffusionOptions =
      py::class_<snap::DiffusionOptionsImpl, snap::DiffusionOptions>(
          m, "DiffusionOptions");

  pyDiffusionOptions.def(py::init<>())
      .def("__repr__",
           [](const snap::DiffusionOptions& a) {
             std::stringstream ss;
             a->report(ss);
             return fmt::format("DiffusionOptions(\n{})", ss.str());
           })
      .ADD_OPTION(double, snap::DiffusionOptionsImpl, nu_iso)
      .ADD_OPTION(double, snap::DiffusionOptionsImpl, kappa_iso);

  auto pyScalarHyperdiffusionOptions =
      py::class_<snap::ScalarHyperdiffusionOptionsImpl,
                 snap::ScalarHyperdiffusionOptions>(
          m, "ScalarHyperdiffusionOptions");

  pyScalarHyperdiffusionOptions.def(py::init<>())
      .def("__repr__",
           [](const snap::ScalarHyperdiffusionOptions& a) {
             std::stringstream ss;
             a->report(ss);
             return fmt::format("ScalarHyperdiffusionOptions(\n{})", ss.str());
           })
      .ADD_OPTION(double, snap::ScalarHyperdiffusionOptionsImpl, damping_time)
      .ADD_OPTION(std::vector<std::string>,
                  snap::ScalarHyperdiffusionOptionsImpl, fields);
}
