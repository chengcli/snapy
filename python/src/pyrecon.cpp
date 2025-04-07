// torch
#include <torch/extension.h>

// fvm
#include <fvm/recon/recon_formatter.hpp>
#include <fvm/recon/reconstruct.hpp>

// python
#include "pyoptions.hpp"

namespace py = pybind11;

void bind_recon(py::module &m) {
  py::class_<canoe::InterpOptions>(m, "InterpOptions")
      .def(py::init<>())
      .def(py::init<std::string>())
      .def(py::init<canoe::ParameterInput, std::string, std::string>())
      .def("__repr__",
           [](const canoe::InterpOptions &a) {
             return fmt::format("InterpOptions{}", a);
           })
      .ADD_OPTION(std::string, canoe::InterpOptions, type)
      .ADD_OPTION(bool, canoe::InterpOptions, scale);

  py::class_<canoe::ReconstructOptions>(m, "ReconstructOptions")
      .def(py::init<>())
      .def(py::init<canoe::ParameterInput, std::string, std::string>())
      .def("__repr__",
           [](const canoe::ReconstructOptions &a) {
             return fmt::format("ReconstructOptions{}", a);
           })
      .ADD_OPTION(bool, canoe::ReconstructOptions, shock)
      .ADD_OPTION(canoe::InterpOptions, canoe::ReconstructOptions, interp);

  ADD_CANOE_MODULE(Reconstruct, ReconstructOptions)
      .def(py::init<>())
      .def(py::init<canoe::ReconstructOptions>())
      .def("forward", &canoe::ReconstructImpl::forward);
}
