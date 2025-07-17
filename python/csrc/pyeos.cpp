// torch
#include <torch/extension.h>

// kintera
#include <kintera/thermo/thermo.hpp>

// snap
#include <snap/eos/eos_formatter.hpp>
#include <snap/eos/equation_of_state.hpp>

// python
#include "pyoptions.hpp"

namespace py = pybind11;

void bind_eos(py::module &m) {
  auto pyEquationOfStateOptions =
      py::class_<snap::EquationOfStateOptions>(m, "EquationOfStateOptions");

  pyEquationOfStateOptions.def(py::init<>())
      .def("__repr__",
           [](const snap::EquationOfStateOptions &a) {
             std::stringstream ss;
             a.report(ss);
             return fmt::format("EquationOfStateOptions(\n{})", ss.str());
           })
      .ADD_OPTION(std::string, snap::EquationOfStateOptions, type)
      .ADD_OPTION(double, snap::EquationOfStateOptions, density_floor)
      .ADD_OPTION(double, snap::EquationOfStateOptions, pressure_floor)
      .ADD_OPTION(bool, snap::EquationOfStateOptions, limiter)
      .ADD_OPTION(kintera::ThermoOptions, snap::EquationOfStateOptions, thermo)
      .ADD_OPTION(snap::CoordinateOptions, snap::EquationOfStateOptions, coord);

  py::class_<snap::EquationOfStateImpl>(m, "EquationOfState")
      .def("__repr__",
           [](const snap::EquationOfStateImpl &a) {
             return fmt::format("EquationOfState(\n{})", a.options);
           })
      .def("nvar", &snap::EquationOfStateImpl::nvar)
      .def("compute", &snap::EquationOfStateImpl::compute)
      .def("forward", &snap::EquationOfStateImpl::forward);
}
