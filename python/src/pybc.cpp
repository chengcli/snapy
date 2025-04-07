// pybind11
#include <pybind11/functional.h>

// torch
#include <torch/extension.h>
#include <torch/nn/modules/container/any.h>

// fvm
#include <fvm/bc/bc_formatter.hpp>
#include <fvm/bc/boundary_condition.hpp>
#include <fvm/bc/internal_boundary.hpp>

// python
#include "pyoptions.hpp"

namespace py = pybind11;

void bind_bc(py::module &m) {
  py::enum_<canoe::BoundaryFace>(m, "BoundaryFace")
      .value("kUnknown", canoe::kUnknown)
      .value("kInnerX1", canoe::kInnerX1)
      .value("kOuterX1", canoe::kOuterX1)
      .value("kInnerX2", canoe::kInnerX2)
      .value("kOuterX2", canoe::kOuterX2)
      .value("kInnerX3", canoe::kInnerX3)
      .value("kOuterX3", canoe::kOuterX3)
      .export_values();

  py::enum_<canoe::BoundaryFlag>(m, "BoundaryFlag")
      .value("kExchange", canoe::BoundaryFlag::kExchange)
      .value("kUser", canoe::BoundaryFlag::kUser)
      .value("kReflect", canoe::BoundaryFlag::kReflect)
      .value("kOutflow", canoe::BoundaryFlag::kOutflow)
      .value("kPeriodic", canoe::BoundaryFlag::kPeriodic)
      .value("kShearPeriodic", canoe::BoundaryFlag::kShearPeriodic)
      .value("kPolar", canoe::BoundaryFlag::kPolar)
      .value("kPolarWedge", canoe::BoundaryFlag::kPolarWedge)
      .export_values();

  py::class_<canoe::BoundaryFuncOptions>(m, "BoundaryFuncOptions")
      .def(py::init<>())
      .def("__repr__",
           [](const canoe::BoundaryFuncOptions &a) {
             return fmt::format("BoundaryFuncOptions{}", a);
           })
      .ADD_OPTION(int, canoe::BoundaryFuncOptions, type)
      .ADD_OPTION(int, canoe::BoundaryFuncOptions, nghost);

  py::class_<canoe::InternalBoundaryOptions>(m, "InternalBoundaryOptions")
      .def(py::init<>())
      .def("__repr__",
           [](const canoe::InternalBoundaryOptions &a) {
             return fmt::format("InternalBoundaryOptions{}", a);
           })
      .ADD_OPTION(int, canoe::InternalBoundaryOptions, nghost)
      .ADD_OPTION(int, canoe::InternalBoundaryOptions, max_iter)
      .ADD_OPTION(double, canoe::InternalBoundaryOptions, solid_density)
      .ADD_OPTION(double, canoe::InternalBoundaryOptions, solid_pressure);

  ADD_CANOE_MODULE(InternalBoundary, InternalBoundaryOptions)
      .def_readonly("options", &canoe::InternalBoundaryImpl::options)
      .def("mark_solid", &canoe::InternalBoundaryImpl::mark_solid)
      .def(
          "rectify_solid",
          [](canoe::InternalBoundaryImpl &self, torch::Tensor solid_in,
             std::vector<canoe::bfunc_t> const &bfuncs) {
            int total_num_flips = 0;
            auto result = self.rectify_solid(solid_in, total_num_flips, bfuncs);
            return std::make_pair(result, total_num_flips);
          },
          py::arg("solid"), py::arg("bfuncs") = std::vector<canoe::bfunc_t>{})
      .def("forward", &canoe::InternalBoundaryImpl::forward);
}
