// torch
#include <torch/extension.h>

// input
#include <input/parameter_input.hpp>

// fvm
#include <fvm/eos/eos_formatter.hpp>
#include <fvm/eos/equation_of_state.hpp>
#include <fvm/thermo/thermodynamics.hpp>

// python
#include "pyoptions.hpp"

namespace py = pybind11;

void bind_eos(py::module &m) {
  py::class_<canoe::EquationOfStateOptions>(m, "EquationOfStateOptions")
      .def(py::init<>())
      .def(py::init<canoe::ParameterInput>())
      .def("__repr__",
           [](const canoe::EquationOfStateOptions &a) {
             return fmt::format("EquationOfStateOptions{}", a);
           })
      .ADD_OPTION(std::string, canoe::EquationOfStateOptions, type)
      .ADD_OPTION(double, canoe::EquationOfStateOptions, density_floor)
      .ADD_OPTION(double, canoe::EquationOfStateOptions, pressure_floor)
      .ADD_OPTION(bool, canoe::EquationOfStateOptions, limiter)
      .ADD_OPTION(canoe::ThermodynamicsOptions, canoe::EquationOfStateOptions,
                  thermo)
      .ADD_OPTION(canoe::CoordinateOptions, canoe::EquationOfStateOptions,
                  coord);

  ADD_CANOE_MODULE(IdealGas, EquationOfStateOptions)
      .def("sound_speed", &canoe::IdealGasImpl::sound_speed)
      .def("prim2cons", &canoe::IdealGasImpl::prim2cons);

  ADD_CANOE_MODULE(ShallowWater, EquationOfStateOptions)
      .def("sound_speed", &canoe::ShallowWaterImpl::sound_speed)
      .def("prim2cons", &canoe::ShallowWaterImpl::prim2cons);
}
