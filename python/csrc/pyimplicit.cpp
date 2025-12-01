// C/C++
#include <sstream>

// torch
#include <torch/extension.h>
#include <torch/nn/modules/container/any.h>

// snap
#include <snap/implicit/implicit.hpp>

// python
#include "pyoptions.hpp"

namespace py = pybind11;

void bind_implicit(py::module &m) {
  auto pyImplicitOptions =
      py::class_<snap::ImplicitOptionsImpl, snap::ImplicitOptions>(
          m, "ImplicitOptions");

  pyImplicitOptions.def(py::init<>())
      .def_static("from_yaml",
                  py::overload_cast<std::string const &, bool>(
                      &snap::ImplicitOptionsImpl::from_yaml),
                  py::arg("filename"), py::arg("verbose") = false)
      .def("__repr__",
           [](const snap::ImplicitOptions &a) {
             std::stringstream ss;
             a->report(ss);
             return fmt::format("ImplicitOptions(\n{})", ss.str());
           })
      .ADD_OPTION(std::string, snap::ImplicitOptionsImpl, type)
      .ADD_OPTION(int, snap::ImplicitOptionsImpl, scheme)
      .ADD_OPTION(snap::ConstGravityOptions, snap::ImplicitOptionsImpl, grav)
      .ADD_OPTION(snap::CoordinateOptions, snap::ImplicitOptionsImpl, coord);

  ADD_SNAP_MODULE(ImplicitHydro, ImplicitOptions);
  ADD_SNAP_MODULE(ImplicitCorrection, ImplicitOptions);
}
