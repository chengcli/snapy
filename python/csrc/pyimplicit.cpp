// torch
#include <torch/extension.h>
#include <torch/nn/modules/container/any.h>

// snap
#include <snap/implicit/implicit_formatter.hpp>
#include <snap/implicit/vertical_implicit.hpp>

// python
#include "pyoptions.hpp"

namespace py = pybind11;

void bind_implicit(py::module &m) {
  auto pyImplicitOptions =
      py::class_<snap::ImplicitOptions>(m, "ImplicitOptions");

  pyImplicitOptions.def(py::init<>())
      .def("__repr__",
           [](const snap::ImplicitOptions &a) {
             return fmt::format("ImplicitOptions{}", a);
           })
      .ADD_OPTION(std::string, snap::ImplicitOptions, type)
      .ADD_OPTION(int, snap::ImplicitOptions, nghost)
      .ADD_OPTION(double, snap::ImplicitOptions, grav)
      .ADD_OPTION(int, snap::ImplicitOptions, scheme)
      .ADD_OPTION(snap::CoordinateOptions, snap::ImplicitOptions, coord)
      .ADD_OPTION(snap::ReconstructOptions, snap::ImplicitOptions, recon);

  ADD_SNAP_MODULE(VerticalImplicit, ImplicitOptions);
}
