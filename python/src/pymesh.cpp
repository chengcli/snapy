// pybind11
#include <pybind11/functional.h>

// torch
#include <torch/extension.h>

// base
#include <globals.h>

// fvm
#include <fvm/mesh/mesh.hpp>
#include <fvm/mesh/mesh_formatter.hpp>
#include <fvm/mesh/meshblock.hpp>

// python
#include "pyoptions.hpp"

namespace py = pybind11;

void bind_mesh(py::module &m) {
  py::class_<canoe::MeshOptions>(m, "MeshOptions")
      .def(py::init<>())
      .def(py::init<canoe::ParameterInput>())
      .def_readwrite("bflags", &canoe::MeshOptions::bflags)
      .def("__repr__",
           [](const canoe::MeshOptions &a) {
             return fmt::format("MeshOptions{}", a);
           })
      .ADD_OPTION(double, canoe::MeshOptions, x1min)
      .ADD_OPTION(double, canoe::MeshOptions, x1max)
      .ADD_OPTION(double, canoe::MeshOptions, x2min)
      .ADD_OPTION(double, canoe::MeshOptions, x2max)
      .ADD_OPTION(double, canoe::MeshOptions, x3min)
      .ADD_OPTION(double, canoe::MeshOptions, x3max)
      .ADD_OPTION(canoe::MeshBlockOptions, canoe::MeshOptions, block)
      .ADD_OPTION(canoe::OctTreeOptions, canoe::MeshOptions, tree);

  py::class_<canoe::OctTreeOptions>(m, "OctTreeOptions")
      .def(py::init<>())
      .def(py::init<canoe::ParameterInput>())
      .def("__repr__",
           [](const canoe::OctTreeOptions &a) {
             return fmt::format("OctTreeOptions{}", a);
           })
      .ADD_OPTION(int, canoe::OctTreeOptions, nb1)
      .ADD_OPTION(int, canoe::OctTreeOptions, nb2)
      .ADD_OPTION(int, canoe::OctTreeOptions, nb3)
      .ADD_OPTION(int, canoe::OctTreeOptions, ndim);

  py::class_<canoe::MeshBlockOptions>(m, "MeshBlockOptions")
      .def(py::init<>())
      .def(py::init<canoe::ParameterInput>())
      .def("__repr__",
           [](const canoe::MeshBlockOptions &a) {
             return fmt::format("MeshBlockOptions{}", a);
           })
      .ADD_OPTION(int, canoe::MeshBlockOptions, nghost)
      .ADD_OPTION(canoe::IntegratorOptions, canoe::MeshBlockOptions, intg)
      .ADD_OPTION(canoe::HydroOptions, canoe::MeshBlockOptions, hydro)
      .ADD_OPTION(canoe::ScalarOptions, canoe::MeshBlockOptions, scalar)
      .ADD_OPTION(std::vector<canoe::BoundaryFlag>, canoe::MeshBlockOptions,
                  bflags);

  torch::python::bind_module<canoe::LogicalLocationImpl>(m, "LogicalLocation")
      .def(py::init<>())
      .def(py::init<int, int64_t, int64_t, int64_t>())
      .def("__repr__",
           [](const canoe::LogicalLocationImpl &a) {
             return fmt::format(
                 "LogicalLocation(level = {}; lx1 = {:b}; lx2 = {:b}; lx3 = "
                 "{:b}",
                 a.level, a.lx1, a.lx2, a.lx3);
           })
      .def("__lt__", &canoe::LogicalLocationImpl::lesser)
      .def("__gt__", &canoe::LogicalLocationImpl::greater)
      .def("__eq__", &canoe::LogicalLocationImpl::equal)
      .def_readwrite("lx1", &canoe::LogicalLocationImpl::lx1)
      .def_readwrite("lx2", &canoe::LogicalLocationImpl::lx2)
      .def_readwrite("lx3", &canoe::LogicalLocationImpl::lx3)
      .def_readwrite("level", &canoe::LogicalLocationImpl::level);

  torch::python::bind_module<canoe::OctTreeNodeImpl>(m, "OctTreeNode")
      .def_readwrite("leaves", &canoe::OctTreeNodeImpl::leaves)
      .def_readwrite("loc", &canoe::OctTreeNodeImpl::loc)
      .def("nleaf", &canoe::OctTreeNodeImpl::nleaf)
      .def("find_node", &canoe::OctTreeNodeImpl::find_node);

  torch::python::bind_module<canoe::OctTreeImpl>(m, "OctTree")
      .def(py::init<>())
      .def(py::init<canoe::OctTreeOptions>())
      //.def("root", [](canoe::OctTreeImpl &self) { return self.root.ptr(); })
      .def("root_level", &canoe::OctTreeImpl::root_level)
      .def("forward", &canoe::OctTreeImpl::forward);

  torch::python::bind_module<canoe::MeshBlockImpl>(m, "MeshBlock")
      .def(py::init<>())
      .def(py::init<canoe::MeshBlockOptions>())
      .def(py::init<canoe::MeshBlockOptions, canoe::LogicalLocation>())
      .def_readwrite("bfuncs", &canoe::MeshBlockImpl::bfuncs)
      .def("nc1", &canoe::MeshBlockImpl::nc1)
      .def("nc2", &canoe::MeshBlockImpl::nc2)
      .def("nc3", &canoe::MeshBlockImpl::nc3)
      .def(
          "part",
          [](canoe::MeshBlockImpl &self, std::tuple<int, int, int> offset,
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
      .def("set_primitives", &canoe::MeshBlockImpl::set_primitives,
           py::arg("hydro_w"), py::arg("scalar_w") = torch::nullopt)
      .def("max_root_time_step", &canoe::MeshBlockImpl::max_root_time_step,
           py::arg("root_level") = 0, py::arg("solid") = torch::nullopt)
      .def("max_time_step", &canoe::MeshBlockImpl::max_root_time_step,
           py::arg("root_level") = 0, py::arg("solid") = torch::nullopt)
      .def("forward", &canoe::MeshBlockImpl::forward, py::arg("dt"),
           py::arg("stage"), py::arg("solid") = torch::nullopt)
      .def("set_uov",
           [](canoe::MeshBlockImpl &self, std::string name, torch::Tensor val) {
             if (self.user_out_var.contains(name)) {
               self.user_out_var[name] = val;
             } else {
               self.user_out_var.insert(name, val);
             }
           })
      .def("var", [](canoe::MeshBlockImpl &self, std::string name) {
        return GET_SHARED(name);
      });

  torch::python::bind_module<canoe::MeshImpl>(m, "Mesh")
      .def(py::init<>(), R"(
          Construct a default Mesh object.
        )")
      .def(py::init<canoe::MeshOptions>(), R"(
          Construct a new Mesh object with the given options.
        )")
      .def_readwrite("start_time", &canoe::MeshImpl::start_time, R"(
          Returns the start time of the simulation.
        )")
      .def_readwrite("current_time", &canoe::MeshImpl::current_time, R"(
          Returns the current time of the simulation.
        )")
      .def_readwrite("tlim", &canoe::MeshImpl::tlim, R"(
          Returns the time limit of the simulation.
        )")
      .def("max_time_step", &canoe::MeshImpl::max_time_step, R"(
          Returns the maximum time step of the simulation.
        )")
      .def("forward", &canoe::MeshImpl::forward, R"(
          Advances the simulation to the desired time.

          Parameters
          ----------
          time : float
              The time step to advance the simulation.
          max_steps : int

          Returns
          -------
          None
        )",
           py::arg("time"), py::arg("max_steps") = -1);
}
