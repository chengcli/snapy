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
  py::class_<snap::VerticalImplicitOptions>(m, "VerticalImplicitOptions")
      .def(py::init<>())
      .def("__repr__",
           [](const snap::VerticalImplicitOptions &a) {
             return fmt::format("VerticalImplicitOptions{}", a);
           })
      .ADD_OPTION(std::string, snap::VerticalImplicitOptions, type)
      .ADD_OPTION(int, snap::VerticalImplicitOptions, batch_size)
      .ADD_OPTION(int, snap::VerticalImplicitOptions, scheme);

  ADD_SNAP_MODULE(VerticalImplicit, VerticalImplicitOptions)
      .def("forward", &snap::VerticalImplicitImpl::forward);
}
