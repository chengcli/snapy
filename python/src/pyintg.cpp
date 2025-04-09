// torch
#include <torch/extension.h>
#include <torch/nn/modules/container/any.h>

// snap
#include <snap/intg/integrator.hpp>
#include <snap/intg/intg_formatter.hpp>

// python
#include "pyoptions.hpp"

namespace py = pybind11;

void bind_intg(py::module &m) {
  py::class_<snap::IntegratorWeight>(m, "IntegratorWeight")
      .def(py::init<>())
      .def("__repr__",
           [](const snap::IntegratorWeight &a) {
             return fmt::format("IntegratorWeight{}", a);
           })
      .ADD_OPTION(double, snap::IntegratorWeight, wght0)
      .ADD_OPTION(double, snap::IntegratorWeight, wght1)
      .ADD_OPTION(double, snap::IntegratorWeight, wght2);

  py::class_<snap::IntegratorOptions>(m, "IntegratorOptions")
      .def(py::init<>())
      .def("__repr__",
           [](const snap::IntegratorOptions &a) {
             return fmt::format("IntegratorOptions{}", a);
           })
      .ADD_OPTION(std::string, snap::IntegratorOptions, type)
      .ADD_OPTION(double, snap::IntegratorOptions, cfl);

  ADD_SNAP_MODULE(Integrator, IntegratorOptions)
      .def_readonly("options", &snap::IntegratorImpl::options)
      .def_readonly("stages", &snap::IntegratorImpl::stages);
}
