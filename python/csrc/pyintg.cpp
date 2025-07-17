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
  auto pyIntegratorWeight =
      py::class_<snap::IntegratorWeight>(m, "IntegratorWeight");

  pyIntegratorWeight.def(py::init<>())
      .def("__repr__",
           [](const snap::IntegratorWeight &a) {
             std::stringstream ss;
             a.report(ss);
             return fmt::format("IntegratorWeight(\n{})", ss.str());
           })
      .ADD_OPTION(double, snap::IntegratorWeight, wght0)
      .ADD_OPTION(double, snap::IntegratorWeight, wght1)
      .ADD_OPTION(double, snap::IntegratorWeight, wght2);

  auto pyIntegratorOptions =
      py::class_<snap::IntegratorOptions>(m, "IntegratorOptions");

  pyIntegratorOptions.def(py::init<>())
      .def("__repr__",
           [](const snap::IntegratorOptions &a) {
             std::stringstream ss;
             a.report(ss);
             return fmt::format("IntegratorOptions(\n{})", ss.str());
           })
      .ADD_OPTION(std::string, snap::IntegratorOptions, type)
      .ADD_OPTION(double, snap::IntegratorOptions, cfl);

  ADD_SNAP_MODULE(Integrator, IntegratorOptions)
      .def_readonly("stages", &snap::IntegratorImpl::stages)
      .def("stop", &snap::IntegratorImpl::stop, py::arg("steps"),
           py::arg("current_time"));
}
