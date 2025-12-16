// torch
#include <torch/extension.h>

// kintera
#include <kintera/thermo/thermo.hpp>

// snap
#include <snap/eos/equation_of_state.hpp>
#include <snap/eos/ideal_gas.hpp>
#include <snap/eos/ideal_moist.hpp>
#include <snap/eos/moist_mixture.hpp>

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

  auto pyEquationOfState =
      py::class_<snap::EquationOfStateImpl, snap::EquationOfState>(
          m, "EquationOfState");

  pyEquationOfState.def(py::init<>())
      .def(py::init<snap::EquationOfStateOptions, torch::nn::Module *>(),
           py::arg("options"), py::arg("phydro") = nullptr)
      .def("__repr__",
           [](const snap::EquationOfStateImpl &a) {
             std::stringstream ss;
             a.options->report(ss);
             return fmt::format("EquationOfState(\n{})", ss.str());
           })
      .def("nvar", &snap::EquationOfStateImpl::nvar)
      .def("compute", &snap::EquationOfStateImpl::compute);

  /*auto pyIdealGas = py::class_<
    snap::IdealGasImpl, snap::EquationOfStateImpl,
    std::shared_ptr<snap::IdealGasImpl>(m, "IdealGas");

  pyIdealGas.def(py::init<>())
      .def("__repr__",
           [](const snap::IdealGasImpl &a) {
             std::stringstream ss;
             a.options->report(ss);
             return fmt::format("IdealGas(\n{})", ss.str());
           })
      .def("nvar", &snap::IdealGasImpl::nvar)
      .def("compute", &snap::IdealGasImpl::compute);

  auto pyIdealMoist = py::class_<
    snap::IdealMoistImpl, snap::EquationOfStateImpl,
    std::shared_ptr<snap::IdealMoistImpl>(m, "IdealMoist");

  pyIdealMoist.def(py::init<>())
      .def("__repr__",
           [](const snap::IdealMoistImpl &a) {
             std::stringstream ss;
             a.options->report(ss);
             return fmt::format("IdealMoist(\n{})", ss.str());
           })
      .def("nvar", &snap::IdealMoistImpl::nvar)
      .def("compute", &snap::IdealMoistImpl::compute);

  auto pyMoistMixture = py::class_<
    snap::MoistMixtureImpl, snap::EquationOfStateImpl,
    std::shared_ptr<snap::MoistMixtureImpl>(m, "MoistMixture");

  pyMoistMixture.def(py::init<>())
      .def("__repr__",
           [](const snap::MoistMixtureImpl &a) {
             std::stringstream ss;
             a.options->report(ss);
             return fmt::format("MoistMixture(\n{})", ss.str());
           })
      .def("nvar", &snap::MoistMixtureImpl::nvar)
      .def("compute", &snap::MoistMixtureImpl::compute);*/
}
