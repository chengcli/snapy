// pybind
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

// base
#include <formatter.hpp>

// dsmc
#include <dsmc/dsmc_initialization.h>
#include <dsmc/io_dsmc.h>

#include <dsmc/dsmc_formatter.hpp>

// python
#include "pyoptions.hpp"

namespace py = pybind11;

void bind_dsmc(py::module &parent) {
  auto m = parent.def_submodule("dsmc", "Python bindings for dsmc module");

  py::class_<CellInitOptions>(m, "CellInitOptions")
      .def(py::init<>())
      .def("__repr__",
           [](const CellInitOptions &a) {
             return fmt::format("CellInitOptions{}", a);
           })
      .ADD_OPTION(double, CellInitOptions, Rho_gas)
      .ADD_OPTION(double, CellInitOptions, T_gas)
      .ADD_OPTION(double, CellInitOptions, Vgasx)
      .ADD_OPTION(double, CellInitOptions, Vgasy)
      .ADD_OPTION(double, CellInitOptions, Vgasz);

  m.def("initialize_uniform", &dsmc_initialization_uniform);
  m.def("output", &dsmc_output);
  m.def("integration", &dsmc_integration);
}
