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
  py::class_<snap::ImplicitOptions>(m, "ImplicitOptions")
      .def(py::init<>())
      .def("__repr__",
           [](const snap::ImplicitOptions &a) {
             return fmt::format("ImplicitOptions{}", a);
           })
      .ADD_OPTION(std::string, snap::ImplicitOptions, type)
      .ADD_OPTION(int, snap::ImplicitOptions, batch_size)
      .ADD_OPTION(int, snap::ImplicitOptions, scheme);

  ADD_SNAP_MODULE(Implicit, ImplicitOptions);
}
