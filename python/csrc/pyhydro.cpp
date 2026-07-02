// C/C++
#include <sstream>

// torch
#include <torch/extension.h>
#include <torch/nn/modules/container/any.h>

// snap
#include <snap/hydro/hydro.hpp>

// python
#include "pyoptions.hpp"

namespace py = pybind11;

void bind_hydro(py::module &m) {
  auto pyHydroOptions =
      py::class_<snap::HydroOptionsImpl, snap::HydroOptions>(m, "HydroOptions");

  pyHydroOptions.def(py::init<>())
      .def_static("from_yaml", &snap::HydroOptionsImpl::from_yaml,
                  py::arg("filename"), py::arg("verbose") = false)
      .def("__repr__",
           [](const snap::HydroOptions &a) {
             std::stringstream ss;
             a->report(ss);
             return fmt::format("HydroOptions(\n{})", ss.str());
           })
      .ADD_OPTION(bool, snap::HydroOptionsImpl, verbose)
      .ADD_OPTION(bool, snap::HydroOptionsImpl, disable_flux_x1)
      .ADD_OPTION(bool, snap::HydroOptionsImpl, disable_flux_x2)
      .ADD_OPTION(bool, snap::HydroOptionsImpl, disable_flux_x3)
      .ADD_OPTION(snap::ConstGravityOptions, snap::HydroOptionsImpl, grav)
      .ADD_OPTION(snap::CoriolisOptions, snap::HydroOptionsImpl, coriolis)
      .ADD_OPTION(snap::DiffusionOptions, snap::HydroOptionsImpl, diffusion)
      .ADD_OPTION(snap::EquationOfStateOptions, snap::HydroOptionsImpl, eos)
      .ADD_OPTION(snap::ReconstructOptions, snap::HydroOptionsImpl, recon1)
      .ADD_OPTION(snap::ReconstructOptions, snap::HydroOptionsImpl, recon23)
      .ADD_OPTION(snap::RiemannSolverOptions, snap::HydroOptionsImpl, riemann)
      .ADD_OPTION(snap::ImplicitOptions, snap::HydroOptionsImpl, icorr);

  ADD_SNAP_MODULE(Hydro, HydroOptions)
      .def(py::init<snap::HydroOptions, torch::nn::Module *>(),
           py::arg("options"), py::arg("block") = nullptr)
      .def("max_time_step", &snap::HydroImpl::max_time_step);
}
