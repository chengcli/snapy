// torch
#include <torch/extension.h>
#include <torch/nn/modules/container/any.h>

// input
#include <input/parameter_input.hpp>

// fvm
#include <fvm/hydro/hydro.hpp>
#include <fvm/hydro/hydro_formatter.hpp>

// python
#include "pyoptions.hpp"

namespace py = pybind11;

void bind_hydro(py::module &m) {
  py::class_<canoe::HydroOptions>(m, "HydroOptions")
      .def(py::init<>())
      .def(py::init<canoe::ParameterInput>())
      .def("__repr__",
           [](const canoe::HydroOptions &a) {
             return fmt::format("HydroOptions{}", a);
           })
      .ADD_OPTION(canoe::ConstGravityOptions, canoe::HydroOptions, grav)
      .ADD_OPTION(canoe::CoriolisOptions, canoe::HydroOptions, coriolis)
      .ADD_OPTION(canoe::EquationOfStateOptions, canoe::HydroOptions, eos)
      .ADD_OPTION(canoe::CoordinateOptions, canoe::HydroOptions, coord)
      .ADD_OPTION(canoe::RiemannSolverOptions, canoe::HydroOptions, riemann)
      .ADD_OPTION(canoe::PrimitiveProjectorOptions, canoe::HydroOptions, proj)
      .ADD_OPTION(canoe::ReconstructOptions, canoe::HydroOptions, recon1)
      .ADD_OPTION(canoe::ReconstructOptions, canoe::HydroOptions, recon23)
      .ADD_OPTION(canoe::InternalBoundaryOptions, canoe::HydroOptions, ib)
      .ADD_OPTION(canoe::VerticalImplicitOptions, canoe::HydroOptions, vic);

  py::class_<canoe::PrimitiveProjectorOptions>(m, "PrimitiveProjectorOptions")
      .def(py::init<>())
      .def("__repr__",
           [](const canoe::PrimitiveProjectorOptions &a) {
             return fmt::format("PrimitiveProjectorOptions{}", a);
           })
      .ADD_OPTION(std::string, canoe::PrimitiveProjectorOptions, type)
      .ADD_OPTION(double, canoe::PrimitiveProjectorOptions, margin);

  ADD_CANOE_MODULE(Hydro, HydroOptions)
      .def("nvar", &canoe::HydroImpl::nvar)
      .def("max_time_step", &canoe::HydroImpl::max_time_step)
      .def("forward", &canoe::HydroImpl::forward);
}
