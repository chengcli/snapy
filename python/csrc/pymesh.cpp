// pybind11
#include <pybind11/functional.h>

// torch
#include <torch/extension.h>

// snap
#include <snap/mesh/mesh_formatter.hpp>
#include <snap/mesh/meshblock.hpp>

// python
#include "pyoptions.hpp"

namespace py = pybind11;

void bind_mesh(py::module &m) {
  auto pyMeshBlockOptions =
      py::class_<snap::MeshBlockOptions>(m, "MeshBlockOptions");

  pyMeshBlockOptions.def(py::init<>())
      .def("__repr__",
           [](const snap::MeshBlockOptions &a) {
             return fmt::format("MeshBlockOptions{}", a);
           })
      .def("from_yaml", &snap::MeshBlockOptions::from_yaml)
      .ADD_OPTION(int, snap::MeshBlockOptions, lx1)
      .ADD_OPTION(int, snap::MeshBlockOptions, lx2)
      .ADD_OPTION(int, snap::MeshBlockOptions, lx3)
      .ADD_OPTION(int, snap::MeshBlockOptions, level)
      .ADD_OPTION(int, snap::MeshBlockOptions, gid)
      .ADD_OPTION(snap::IntegratorOptions, snap::MeshBlockOptions, intg)
      .ADD_OPTION(snap::HydroOptions, snap::MeshBlockOptions, hydro)
      .ADD_OPTION(snap::ScalarOptions, snap::MeshBlockOptions, scalar)
      .ADD_OPTION(std::vector<bcfunc_t>, snap::MeshBlockOptions, bfuncs);

  ADD_SNAP_MODULE(MeshBlock, MeshBlockOptions)
      .def(
          "part",
          [](snap::MeshBlockImpl &self, std::tuple<int, int, int> offset,
             bool exterior, int extend_x1, int extend_x2, int extend_x3) {
            auto result =
                self.part(offset, exterior, extend_x1, extend_x2, extend_x3);
            py::tuple index_spec(result.size());
            for (size_t i = 0; i < result.size(); ++i) {
              auto s = result[i].slice();
              index_spec[i] =
                  py::slice(s.start().expect_int(), s.stop().expect_int(),
                            s.step().expect_int());
            }
            return index_spec;
          },
          py::arg("offset"), py::arg("exterior") = false,
          py::arg("extend_x1") = 0, py::arg("extend_x2") = 0,
          py::arg("extend_x3") = 0)
      .def("initialize", &snap::MeshBlockImpl::initialize, py::arg("hydro_w"),
           py::arg("scalar_w") = torch::Tensor())
      .def("max_time_step", &snap::MeshBlockImpl::max_time_step,
           py::arg("solid") = torch::Tensor())
      .def("set_uov",
           [](snap::MeshBlockImpl &self, std::string name, torch::Tensor val) {
             if (self.user_out_var.contains(name)) {
               self.user_out_var[name] = val;
             } else {
               self.user_out_var.insert(name, val);
             }
           });
}
