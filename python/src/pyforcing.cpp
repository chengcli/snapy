// torch
#include <torch/extension.h>
#include <torch/nn/modules/container/any.h>

// input
#include <input/parameter_input.hpp>

// fvm
#include <fvm/forcing/forcing.hpp>
#include <fvm/forcing/forcing_formatter.hpp>

// python
#include "pyoptions.hpp"

namespace py = pybind11;

void bind_forcing(py::module &m) {
  py::class_<canoe::ConstGravityOptions>(m, "ConstGravityOptions")
      .def(py::init<>())
      .def(py::init<canoe::ParameterInput>())
      .def("__repr__",
           [](const canoe::ConstGravityOptions &a) {
             return fmt::format("ConstGravityOptions{}", a);
           })
      .ADD_OPTION(double, canoe::ConstGravityOptions, grav1)
      .ADD_OPTION(double, canoe::ConstGravityOptions, grav2)
      .ADD_OPTION(double, canoe::ConstGravityOptions, grav3);

  py::class_<canoe::CoriolisOptions>(m, "CoriolisOptions")
      .def(py::init<>())
      .def(py::init<canoe::ParameterInput>())
      .def("__repr__",
           [](const canoe::CoriolisOptions &a) {
             return fmt::format("CoriolisOptions{}", a);
           })
      .ADD_OPTION(double, canoe::CoriolisOptions, omega1)
      .ADD_OPTION(double, canoe::CoriolisOptions, omega2)
      .ADD_OPTION(double, canoe::CoriolisOptions, omega3)
      .ADD_OPTION(double, canoe::CoriolisOptions, omegax)
      .ADD_OPTION(double, canoe::CoriolisOptions, omegay)
      .ADD_OPTION(double, canoe::CoriolisOptions, omegaz);
}
