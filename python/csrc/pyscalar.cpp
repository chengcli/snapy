// pybind11
#include <pybind11/stl.h>

// C/C++
#include <sstream>

// torch
#include <torch/extension.h>

// snap
#include <snap/recon/reconstruct.hpp>
#include <snap/scalar/scalar.hpp>

// python
#include "pyoptions.hpp"

namespace py = pybind11;

void bind_scalar(py::module& m) {
  auto pyScalarOptions =
      py::class_<snap::ScalarOptionsImpl, snap::ScalarOptions>(m,
                                                               "ScalarOptions");

  pyScalarOptions.def(py::init<>())
      .def("__repr__",
           [](const snap::ScalarOptions& a) {
             std::stringstream ss;
             a->report(ss);
             return fmt::format("ScalarOptions(\n{})", ss.str());
           })
      .ADD_OPTION(bool, snap::ScalarOptionsImpl, verbose)
      .ADD_OPTION(int, snap::ScalarOptionsImpl, nvar)
      .ADD_OPTION(std::vector<std::string>, snap::ScalarOptionsImpl, names)
      .ADD_OPTION(double, snap::ScalarOptionsImpl, upper_bound)
      .ADD_OPTION(snap::ReconstructOptions, snap::ScalarOptionsImpl, recon)
      .ADD_OPTION(snap::RiemannSolverOptions, snap::ScalarOptionsImpl, riemann);

  ADD_SNAP_MODULE(Scalar, ScalarOptions)
      .def(py::init<snap::ScalarOptions, torch::nn::Module*>(),
           py::arg("options"), py::arg("block") = nullptr);
}
