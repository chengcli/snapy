// C/C++
#include <sstream>

// torch
#include <torch/extension.h>
#include <torch/nn/modules/container/any.h>

// snap
#include <snap/hydro/hydro.hpp>
#include <snap/hydro/hydro_formatter.hpp>

// python
#include "pyoptions.hpp"

namespace py = pybind11;

void bind_hydro(py::module &m) {
  auto pyHydroOptions = py::class_<snap::HydroOptions>(m, "HydroOptions");

  pyHydroOptions.def(py::init<>())
      .def("__repr__",
           [](const snap::HydroOptions &a) {
             std::stringstream ss;
             a.report(ss);
             return fmt::format("HydroOptions(\n{})", ss.str());
           })
      .def_static(
          "from_yaml",
          [](const std::string &filename, const snap::DistributeInfo &dist) {
            return snap::HydroOptions::from_yaml(filename, dist);
          },
          py::arg("filename"), py::arg("dist") = snap::DistributeInfo())
      .ADD_OPTION(snap::ConstGravityOptions, snap::HydroOptions, grav)
      .ADD_OPTION(snap::CoriolisOptions, snap::HydroOptions, coriolis)
      .ADD_OPTION(snap::DiffusionOptions, snap::HydroOptions, visc)
      .ADD_OPTION(snap::CoordinateOptions, snap::HydroOptions, coord)
      .ADD_OPTION(snap::EquationOfStateOptions, snap::HydroOptions, eos)
      .ADD_OPTION(snap::PrimitiveProjectorOptions, snap::HydroOptions, proj)
      .ADD_OPTION(snap::ReconstructOptions, snap::HydroOptions, recon1)
      .ADD_OPTION(snap::ReconstructOptions, snap::HydroOptions, recon23)
      .ADD_OPTION(snap::RiemannSolverOptions, snap::HydroOptions, riemann)
      .ADD_OPTION(snap::InternalBoundaryOptions, snap::HydroOptions, ib)
      .ADD_OPTION(snap::ImplicitOptions, snap::HydroOptions, imp);

  auto pyPrimitiveProjectorOptions =
      py::class_<snap::PrimitiveProjectorOptions>(m,
                                                  "PrimitiveProjectorOptions");

  pyPrimitiveProjectorOptions.def(py::init<>())
      .def("__repr__",
           [](const snap::PrimitiveProjectorOptions &a) {
             std::stringstream ss;
             a.report(ss);
             return fmt::format("PrimitiveProjectorOptions(\n{})", ss.str());
           })
      .ADD_OPTION(std::string, snap::PrimitiveProjectorOptions, type)
      .ADD_OPTION(double, snap::PrimitiveProjectorOptions, margin)
      .ADD_OPTION(int, snap::PrimitiveProjectorOptions, nghost)
      .ADD_OPTION(double, snap::PrimitiveProjectorOptions, grav)
      .ADD_OPTION(double, snap::PrimitiveProjectorOptions, Rd);

  ADD_SNAP_MODULE(Hydro, HydroOptions)
      .def("max_time_step", &snap::HydroImpl::max_time_step)
      .def(
          "get_eos", [](snap::HydroImpl &self) { return self.peos; },
          py::return_value_policy::reference_internal);
}
