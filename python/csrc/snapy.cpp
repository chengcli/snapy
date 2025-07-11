// example from:
// (1) https://pytorch.org/tutorials/advanced/cpp_extension.html
// (2) torch/utils/cpp_extension.py
// (3) torch/csrc/api/include/torch/python.h
#include <torch/extension.h>

// snap
#include <snap/snap.h>

#include <snap/input/parameter_input.hpp>

// python
#include "pyoptions.hpp"

namespace py = pybind11;

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
void bind_forcing(py::module &);
void bind_implicit(py::module &);
void bind_intg(py::module &);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.attr("__name__") = "snap";
  m.doc() = "Python bindings for snap";

  py::enum_<snap::Index>(m, "index")
      .value("idn", snap::Index::IDN)
      .value("ivx", snap::Index::IVX)
      .value("ivy", snap::Index::IVY)
      .value("ivz", snap::Index::IVZ)
      .value("ipr", snap::Index::IPR)
      .value("icy", snap::Index::ICY);

  bind_bc(m);
  bind_coord(m);
  bind_eos(m);
  bind_hydro(m);
  bind_recon(m);
  bind_riemann(m);
  bind_output(m);
  bind_forcing(m);
  bind_implicit(m);
  bind_intg(m);
  bind_mesh(m);
  // bind_scalar(m);
}
