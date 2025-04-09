// torch
#include <torch/extension.h>

// snap
#include <snap/eos/eos_formatter.hpp>
#include <snap/eos/equation_of_state.hpp>
#include <snap/input/parameter_input.hpp>
#include <snap/thermo/thermodynamics.hpp>

// python
#include "pyoptions.hpp"

namespace py = pybind11;

void bind_eos(py::module &m) {
  py::class_<snap::EquationOfStateOptions>(m, "EquationOfStateOptions")
      .def(py::init<>())
      .def(py::init<snap::ParameterInput>())
      .def("__repr__",
           [](const snap::EquationOfStateOptions &a) {
             return fmt::format("EquationOfStateOptions{}", a);
           })
      .ADD_OPTION(std::string, snap::EquationOfStateOptions, type)
      .ADD_OPTION(double, snap::EquationOfStateOptions, density_floor)
      .ADD_OPTION(double, snap::EquationOfStateOptions, pressure_floor)
      .ADD_OPTION(bool, snap::EquationOfStateOptions, limiter)
      .ADD_OPTION(snap::ThermodynamicsOptions, snap::EquationOfStateOptions,
                  thermo)
      .ADD_OPTION(snap::CoordinateOptions, snap::EquationOfStateOptions, coord);

  ADD_SNAP_MODULE(IdealGas, EquationOfStateOptions)
      .def("sound_speed", &snap::IdealGasImpl::sound_speed)
      .def("prim2cons", &snap::IdealGasImpl::prim2cons);

  ADD_SNAP_MODULE(ShallowWater, EquationOfStateOptions)
      .def("sound_speed", &snap::ShallowWaterImpl::sound_speed)
      .def("prim2cons", &snap::ShallowWaterImpl::prim2cons);
}
