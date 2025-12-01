// torch
#include <torch/extension.h>

// snap
#include <snap/recon/reconstruct.hpp>

// python
#include "pyoptions.hpp"

namespace py = pybind11;

void bind_recon(py::module &m) {
  auto pyInterpOptions =
      py::class_<snap::InterpOptionsImpl, snap::InterpOptions>(m,
                                                               "InterpOptions");

  pyInterpOptions.def(py::init<>())
      .def(py::init<std::string>())
      .def("__repr__",
           [](const snap::InterpOptions &a) {
             std::stringstream ss;
             a->report(ss);
             return fmt::format("InterpOptions(\n{})", ss.str());
           })
      .ADD_OPTION(std::string, snap::InterpOptionsImpl, type)
      .ADD_OPTION(bool, snap::InterpOptionsImpl, scale);

  auto pyReconstructOptions =
      py::class_<snap::ReconstructOptionsImpl, snap::ReconstructOptions>(
          m, "ReconstructOptions");

  pyReconstructOptions.def(py::init<>())
      .def("__repr__",
           [](const snap::ReconstructOptions &a) {
             std::stringstream ss;
             a->report(ss);
             return fmt::format("ReconstructOptions(\n{})", ss.str());
           })
      .ADD_OPTION(bool, snap::ReconstructOptionsImpl, shock)
      .ADD_OPTION(snap::InterpOptions, snap::ReconstructOptionsImpl, interp);

  ADD_SNAP_MODULE(Reconstruct, ReconstructOptions);
}
