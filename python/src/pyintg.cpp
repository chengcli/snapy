// torch
#include <torch/extension.h>
#include <torch/nn/modules/container/any.h>

// fvm
#include <fvm/intg/integrator.hpp>
#include <fvm/intg/intg_formatter.hpp>

// python
#include "pyoptions.hpp"

namespace py = pybind11;

void bind_intg(py::module &m) {
  py::class_<canoe::IntegratorWeight>(m, "IntegratorWeight")
      .def(py::init<>())
      .def("__repr__",
           [](const canoe::IntegratorWeight &a) {
             return fmt::format("IntegratorWeight{}", a);
           })
      .ADD_OPTION(double, canoe::IntegratorWeight, wght0)
      .ADD_OPTION(double, canoe::IntegratorWeight, wght1)
      .ADD_OPTION(double, canoe::IntegratorWeight, wght2);

  py::class_<canoe::IntegratorOptions>(m, "IntegratorOptions")
      .def(py::init<>())
      .def("__repr__",
           [](const canoe::IntegratorOptions &a) {
             return fmt::format("IntegratorOptions{}", a);
           })
      .ADD_OPTION(std::string, canoe::IntegratorOptions, type)
      .ADD_OPTION(double, canoe::IntegratorOptions, cfl);

  ADD_CANOE_MODULE(Integrator, IntegratorOptions)
      .def_readonly("options", &canoe::IntegratorImpl::options)
      .def_readonly("stages", &canoe::IntegratorImpl::stages);
}
