// torch
#include <torch/extension.h>

// snap
#include <snap/recon/recon_formatter.hpp>
#include <snap/recon/reconstruct.hpp>

// python
#include "pyoptions.hpp"

namespace py = pybind11;

void bind_recon(py::module &m) {
  py::class_<snap::InterpOptions>(m, "InterpOptions")
      .def(py::init<>())
      .def(py::init<std::string>())
      .def(py::init<snap::ParameterInput, std::string, std::string>())
      .def("__repr__",
           [](const snap::InterpOptions &a) {
             return fmt::format("InterpOptions{}", a);
           })
      .ADD_OPTION(std::string, snap::InterpOptions, type)
      .ADD_OPTION(bool, snap::InterpOptions, scale);

  py::class_<snap::ReconstructOptions>(m, "ReconstructOptions")
      .def(py::init<>())
      .def(py::init<snap::ParameterInput, std::string, std::string>())
      .def("__repr__",
           [](const snap::ReconstructOptions &a) {
             return fmt::format("ReconstructOptions{}", a);
           })
      .ADD_OPTION(bool, snap::ReconstructOptions, shock)
      .ADD_OPTION(snap::InterpOptions, snap::ReconstructOptions, interp);

  ADD_SNAP_MODULE(Reconstruct, ReconstructOptions)
      .def(py::init<>())
      .def(py::init<snap::ReconstructOptions>())
      .def("forward", &snap::ReconstructImpl::forward);
}
