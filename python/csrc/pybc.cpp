// pybind11
#include <pybind11/functional.h>

// torch
#include <torch/extension.h>
#include <torch/nn/modules/container/any.h>

// snap
#include <snap/bc/bc.hpp>
#include <snap/bc/bc_formatter.hpp>
#include <snap/bc/internal_boundary.hpp>

// python
#include "pyoptions.hpp"

namespace py = pybind11;

void bind_bc(py::module &m) {
  py::enum_<snap::BoundaryFace>(m, "BoundaryFace")
      .value("kUnknown", snap::kUnknown)
      .value("kInnerX1", snap::kInnerX1)
      .value("kOuterX1", snap::kOuterX1)
      .value("kInnerX2", snap::kInnerX2)
      .value("kOuterX2", snap::kOuterX2)
      .value("kInnerX3", snap::kInnerX3)
      .value("kOuterX3", snap::kOuterX3)
      .export_values();

  auto pyBoundaryFunctionOptions =
      py::class_<snap::BoundaryFuncOptions>(m, "BoundaryFuncOptions");

  pyBoundaryFunctionOptions.def(py::init<>())
      .def("__repr__",
           [](const snap::BoundaryFuncOptions &a) {
             std::stringstream ss;
             a.report(ss);
             return fmt::format("BoundaryFuncOptions(\n{})", ss.str());
           })
      .ADD_OPTION(int, snap::BoundaryFuncOptions, type)
      .ADD_OPTION(int, snap::BoundaryFuncOptions, nghost);

  auto pyInternalBoundaryOptions =
      py::class_<snap::InternalBoundaryOptions>(m, "InternalBoundaryOptions");

  pyInternalBoundaryOptions.def(py::init<>())
      .def("__repr__",
           [](const snap::InternalBoundaryOptions &a) {
             std::stringstream ss;
             a.report(ss);
             return fmt::format("InternalBoundaryOptions(\n{})", ss.str());
           })
      .ADD_OPTION(int, snap::InternalBoundaryOptions, nghost)
      .ADD_OPTION(int, snap::InternalBoundaryOptions, max_iter)
      .ADD_OPTION(double, snap::InternalBoundaryOptions, solid_density)
      .ADD_OPTION(double, snap::InternalBoundaryOptions, solid_pressure);

  ADD_SNAP_MODULE(InternalBoundary, InternalBoundaryOptions)
      .def("mark_solid", &snap::InternalBoundaryImpl::mark_solid)
      .def(
          "rectify_solid",
          [](snap::InternalBoundaryImpl &self, torch::Tensor solid_in,
             std::vector<bcfunc_t> const &bfuncs) {
            int total_num_flips = 0;
            auto result = self.rectify_solid(solid_in, total_num_flips, bfuncs);
            return std::make_pair(result, total_num_flips);
          },
          py::arg("solid"), py::arg("bfuncs") = std::vector<bcfunc_t>{});
}
