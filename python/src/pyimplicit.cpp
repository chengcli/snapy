// torch
#include <torch/extension.h>
#include <torch/nn/modules/container/any.h>

// fvm
#include <fvm/implicit/implicit_formatter.hpp>
#include <fvm/implicit/vertical_implicit.hpp>

// python
#include "pyoptions.hpp"

namespace py = pybind11;

void bind_implicit(py::module &m) {
  py::class_<canoe::VerticalImplicitOptions>(m, "VerticalImplicitOptions")
      .def(py::init<>())
      .def("__repr__",
           [](const canoe::VerticalImplicitOptions &a) {
             return fmt::format("VerticalImplicitOptions{}", a);
           })
      .ADD_OPTION(std::string, canoe::VerticalImplicitOptions, type)
      .ADD_OPTION(int, canoe::VerticalImplicitOptions, batch_size)
      .ADD_OPTION(int, canoe::VerticalImplicitOptions, scheme);

  ADD_CANOE_MODULE(VerticalImplicit, VerticalImplicitOptions)
      .def("forward", &canoe::VerticalImplicitImpl::forward);
}
