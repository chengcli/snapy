// example from:
// (1) https://pytorch.org/tutorials/advanced/cpp_extension.html
// (2) torch/utils/cpp_extension.py
// (3) torch/csrc/api/include/torch/python.h
#include <torch/extension.h>

// spdlog
#include <spdlog/spdlog.h>

// base
#include <fvm/index.h>
#include <region_size.h>

#include <formatter.hpp>
#include <input/parameter_input.hpp>

// python
#include "pyoptions.hpp"

namespace py = pybind11;

void start_logging(int argc, char **argv);
void bind_bc(py::module &);
void bind_mesh(py::module &);
void bind_hydro(py::module &);
void bind_scalar(py::module &);
void bind_eos(py::module &);
void bind_coord(py::module &);
void bind_recon(py::module &);
void bind_riemann(py::module &);
void bind_output(py::module &);
void bind_dsmc(py::module &);
void bind_thermo(py::module &);
void bind_forcing(py::module &);
void bind_implicit(py::module &);
void bind_intg(py::module &);
void bind_speos(py::module &);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.attr("__name__") = "canoe";
  m.doc() = "Python bindings for canoe";

  m.def("start_logging", [](std::string args) {
    int argc = 1;

    // Create a char** array from the vector of strings
    std::vector<const char *> argv(argc);
    argv[0] = args.c_str();

    // Call the original start_logging function
    start_logging(1, const_cast<char **>(argv.data()));
  });

  py::class_<canoe::ParameterInputImpl,
             std::shared_ptr<canoe::ParameterInputImpl>>(m, "ParameterInput")
      .def(py::init<>())
      .def("GetOrAddInteger", &canoe::ParameterInputImpl::GetOrAddInteger)
      .def("DoesParameterExist",
           &canoe::ParameterInputImpl::DoesParameterExist);

  py::class_<canoe::RegionSize>(m, "RegionSize")
      .def(py::init<>())
      .def("__repr__",
           [](const canoe::RegionSize &a) {
             return fmt::format("RegionSize{}", a);
           })
      .def("size", &canoe::RegionSize::size)
      .def("linear_index", &canoe::RegionSize::linear_index)
      .ADD_OPTION(double, canoe::RegionSize, x1min)
      .ADD_OPTION(double, canoe::RegionSize, x1max)
      .ADD_OPTION(double, canoe::RegionSize, x2min)
      .ADD_OPTION(double, canoe::RegionSize, x2max)
      .ADD_OPTION(double, canoe::RegionSize, x3min)
      .ADD_OPTION(double, canoe::RegionSize, x3max)
      .ADD_OPTION(double, canoe::RegionSize, x1rat)
      .ADD_OPTION(double, canoe::RegionSize, x2rat)
      .ADD_OPTION(double, canoe::RegionSize, x3rat)
      .ADD_OPTION(int, canoe::RegionSize, nx1)
      .ADD_OPTION(int, canoe::RegionSize, nx2)
      .ADD_OPTION(int, canoe::RegionSize, nx3);

  py::enum_<canoe::index>(m, "index")
      .value("idn", canoe::index::IDN)
      .value("ivx", canoe::index::IVX)
      .value("ivy", canoe::index::IVY)
      .value("ivz", canoe::index::IVZ)
      .value("ipr", canoe::index::IPR)
      .value("icy", canoe::index::ICY);

  bind_bc(m);
  bind_mesh(m);
  bind_hydro(m);
  bind_eos(m);
  bind_coord(m);
  bind_recon(m);
  bind_riemann(m);
  bind_output(m);
  bind_thermo(m);
  bind_forcing(m);
  bind_implicit(m);
  bind_intg(m);
  bind_speos(m);
  // bind_dsmc(m);
  // bind_scalar(m);
}
