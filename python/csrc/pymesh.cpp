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
             std::stringstream ss;
             a.report(ss);
             return fmt::format("MeshBlockOptions(\n{})", ss.str());
           })
      .def("from_yaml", &snap::MeshBlockOptions::from_yaml, py::arg("filename"),
           py::arg("dist") = snap::DistributeInfo())
      .def(
          "set_bfunc",
          [&](snap::MeshBlockOptions &self, int dx3, int dx2, int dx1,
              py::object func_obj) {
            bcfunc_t func;
            if (func_obj.is_none()) {
              func = nullptr;
            } else {
              py::function f = py::cast<py::function>(func_obj);
              func = [f](torch::Tensor const &var, int id,
                         snap::BoundaryFuncOptions op) {
                py::gil_scoped_acquire gil;
                f(var, id, op);
              };
            }

            if (self.bfuncs().empty()) {
              throw std::runtime_error(
                  "Cannot set boundary function when bfuncs is empty.");
            } else if (self.bfuncs().size() == 2) {
              if (dx3 != 0 || dx2 != 0) {
                throw std::runtime_error(
                    "Only dx1 can be non-zero when bfuncs has size 2.");
              }
            } else if (self.bfuncs().size() == 4) {
              if (dx3 != 0) {
                throw std::runtime_error(
                    "Only dx1 and dx2 can be non-zero when bfuncs has size 4.");
              }
            } else if (self.bfuncs().size() != 6) {
              throw std::runtime_error(
                  "bfuncs must have size 2, 4, or 6 to set boundary "
                  "functions.");
            }

            if (dx3 == 0 && dx2 == 0 && dx1 == -1) {
              self.bfuncs()[0] = func;
            } else if (dx3 == 0 && dx2 == 0 && dx1 == 1) {
              self.bfuncs()[1] = func;
            } else if (dx3 == 0 && dx2 == -1 && dx1 == 0) {
              self.bfuncs()[2] = func;
            } else if (dx3 == 0 && dx2 == 1 && dx1 == 0) {
              self.bfuncs()[3] = func;
            } else if (dx3 == -1 && dx2 == 0 && dx1 == 0) {
              self.bfuncs()[4] = func;
            } else if (dx3 == 1 && dx2 == 0 && dx1 == 0) {
              self.bfuncs()[5] = func;
            }
          },
          py::arg("dx3"), py::arg("dx2"), py::arg("dx1"), py::arg("func"))
      .ADD_OPTION(snap::DistributeInfo, snap::MeshBlockOptions, dist)
      .ADD_OPTION(snap::IntegratorOptions, snap::MeshBlockOptions, intg)
      .ADD_OPTION(snap::HydroOptions, snap::MeshBlockOptions, hydro)
      .ADD_OPTION(snap::ScalarOptions, snap::MeshBlockOptions, scalar)
      .ADD_OPTION(std::vector<bcfunc_t>, snap::MeshBlockOptions, bfuncs);

  ADD_SNAP_MODULE(MeshBlock, MeshBlockOptions)
      .def(
          "forward",
          [](snap::MeshBlockImpl &self, double dt, int stage, py::dict vars) {
            std::map<std::string, torch::Tensor> native;
            for (auto &kv : vars) {
              std::string key = py::cast<std::string>(kv.first);
              if (!kv.second.is_none()) {
                native[key] = py::cast<torch::Tensor>(kv.second);
              } else {
                native[key] = torch::Tensor();
              }
            }
            return self.forward(dt, stage, native);
          },
          py::arg("dt"), py::arg("stage"), py::arg("vars"))
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
      .def("initialize", &snap::MeshBlockImpl::initialize)
      .def("max_time_step",
           [](snap::MeshBlockImpl &self, py::dict vars) {
             std::map<std::string, torch::Tensor> native;
             for (auto &kv : vars) {
               std::string key = py::cast<std::string>(kv.first);
               if (!kv.second.is_none()) {
                 native[key] = py::cast<torch::Tensor>(kv.second);
               } else {
                 native[key] = torch::Tensor();
               }
             }
             return self.max_time_step(native);
           })
      .def("set_uov", [](snap::MeshBlockImpl &self, std::string name,
                         torch::Tensor val) { self.user_out_var[name] = val; });
}
