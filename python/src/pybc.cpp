// pybind11
#include <pybind11/functional.h>

// torch
#include <torch/extension.h>
#include <torch/nn/modules/container/any.h>

// snap
#include <snap/bc/bc_formatter.hpp>
#include <snap/bc/boundary_condition.hpp>
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

  py::enum_<snap::BoundaryFlag>(m, "BoundaryFlag")
      .value("kExchange", snap::BoundaryFlag::kExchange)
      .value("kUser", snap::BoundaryFlag::kUser)
      .value("kReflect", snap::BoundaryFlag::kReflect)
      .value("kOutflow", snap::BoundaryFlag::kOutflow)
      .value("kPeriodic", snap::BoundaryFlag::kPeriodic)
      .value("kShearPeriodic", snap::BoundaryFlag::kShearPeriodic)
      .value("kPolar", snap::BoundaryFlag::kPolar)
      .value("kPolarWedge", snap::BoundaryFlag::kPolarWedge)
      .export_values();

  py::class_<snap::BoundaryFuncOptions>(m, "BoundaryFuncOptions")
      .def(py::init<>())
      .def("__repr__",
           [](const snap::BoundaryFuncOptions &a) {
             return fmt::format("BoundaryFuncOptions{}", a);
           })
      .ADD_OPTION(int, snap::BoundaryFuncOptions, type)
      .ADD_OPTION(int, snap::BoundaryFuncOptions, nghost);

  py::class_<snap::InternalBoundaryOptions>(m, "InternalBoundaryOptions")
      .def(py::init<>())
      .def("__repr__",
           [](const snap::InternalBoundaryOptions &a) {
             return fmt::format("InternalBoundaryOptions{}", a);
           })
      .ADD_OPTION(int, snap::InternalBoundaryOptions, nghost)
      .ADD_OPTION(int, snap::InternalBoundaryOptions, max_iter)
      .ADD_OPTION(double, snap::InternalBoundaryOptions, solid_density)
      .ADD_OPTION(double, snap::InternalBoundaryOptions, solid_pressure);

  ADD_SNAP_MODULE(InternalBoundary, InternalBoundaryOptions)
      .def_readonly("options", &snap::InternalBoundaryImpl::options)
      .def("mark_solid", &snap::InternalBoundaryImpl::mark_solid)
      .def(
          "rectify_solid",
          [](snap::InternalBoundaryImpl &self, torch::Tensor solid_in,
             std::vector<snap::bfunc_t> const &bfuncs) {
            int total_num_flips = 0;
            auto result = self.rectify_solid(solid_in, total_num_flips, bfuncs);
            return std::make_pair(result, total_num_flips);
          },
          py::arg("solid"), py::arg("bfuncs") = std::vector<snap::bfunc_t>{})
      .def("forward", &snap::InternalBoundaryImpl::forward);
}
