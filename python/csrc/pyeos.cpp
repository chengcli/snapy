// torch
#include <torch/extension.h>

// kintera
#include <kintera/thermo/thermo.hpp>

// snap
#include <snap/eos/equation_of_state.hpp>

// python
#include "pyoptions.hpp"

namespace py = pybind11;

void bind_eos(py::module &m) {
  auto pyEquationOfStateOptions =
      py::class_<snap::EquationOfStateOptionsImpl,
                 snap::EquationOfStateOptions>(m, "EquationOfStateOptions");

  pyEquationOfStateOptions.def(py::init<>())
      .def_static("from_yaml", &snap::EquationOfStateOptionsImpl::from_yaml,
                  py::arg("filename"), py::arg("verobse") = false)
      .def("__repr__",
           [](const snap::EquationOfStateOptions &a) {
             std::stringstream ss;
             a->report(ss);
             return fmt::format("EquationOfStateOptions(\n{})", ss.str());
           })
      .ADD_OPTION(std::string, snap::EquationOfStateOptionsImpl, type)
      .ADD_OPTION(double, snap::EquationOfStateOptionsImpl, density_floor)
      .ADD_OPTION(double, snap::EquationOfStateOptionsImpl, pressure_floor)
      .ADD_OPTION(double, snap::EquationOfStateOptionsImpl, temperature_floor)
      .ADD_OPTION(bool, snap::EquationOfStateOptionsImpl, limiter)
      .ADD_OPTION(bool, snap::EquationOfStateOptionsImpl, verbose)
      .ADD_OPTION(std::string, snap::EquationOfStateOptionsImpl, eos_file)
      .ADD_OPTION(kintera::ThermoOptions, snap::EquationOfStateOptionsImpl,
                  thermo);

  py::class_<snap::EquationOfStateImpl, snap::EquationOfState>(
      m, "EquationOfState")
      .def("__repr__",
           [](const snap::EquationOfStateImpl &a) {
             std::stringstream ss;
             a.options->report(ss);
             return fmt::format("EquationOfState(\n{})", ss.str());
           })
      .def("nvar", &snap::EquationOfStateImpl::nvar)
      .def("compute", &snap::EquationOfStateImpl::compute)
      .def("forward", &snap::EquationOfStateImpl::forward);
}
