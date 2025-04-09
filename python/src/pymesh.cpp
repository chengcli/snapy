// pybind11
#include <pybind11/functional.h>

// torch
#include <torch/extension.h>

// snap
#include <snap/mesh/mesh.hpp>
#include <snap/mesh/mesh_formatter.hpp>
#include <snap/mesh/meshblock.hpp>

// python
#include "pyoptions.hpp"

namespace py = pybind11;

void bind_mesh(py::module &m) {
  py::class_<snap::MeshOptions>(m, "MeshOptions")
      .def(py::init<>())
      .def(py::init<snap::ParameterInput>())
      .def_readwrite("bflags", &snap::MeshOptions::bflags)
      .def("__repr__",
           [](const snap::MeshOptions &a) {
             return fmt::format("MeshOptions{}", a);
           })
      .ADD_OPTION(double, snap::MeshOptions, x1min)
      .ADD_OPTION(double, snap::MeshOptions, x1max)
      .ADD_OPTION(double, snap::MeshOptions, x2min)
      .ADD_OPTION(double, snap::MeshOptions, x2max)
      .ADD_OPTION(double, snap::MeshOptions, x3min)
      .ADD_OPTION(double, snap::MeshOptions, x3max)
      .ADD_OPTION(snap::MeshBlockOptions, snap::MeshOptions, block)
      .ADD_OPTION(snap::OctTreeOptions, snap::MeshOptions, tree);

  py::class_<snap::OctTreeOptions>(m, "OctTreeOptions")
      .def(py::init<>())
      .def(py::init<snap::ParameterInput>())
      .def("__repr__",
           [](const snap::OctTreeOptions &a) {
             return fmt::format("OctTreeOptions{}", a);
           })
      .ADD_OPTION(int, snap::OctTreeOptions, nb1)
      .ADD_OPTION(int, snap::OctTreeOptions, nb2)
      .ADD_OPTION(int, snap::OctTreeOptions, nb3)
      .ADD_OPTION(int, snap::OctTreeOptions, ndim);

  py::class_<snap::MeshBlockOptions>(m, "MeshBlockOptions")
      .def(py::init<>())
      .def(py::init<snap::ParameterInput>())
      .def("__repr__",
           [](const snap::MeshBlockOptions &a) {
             return fmt::format("MeshBlockOptions{}", a);
           })
      .ADD_OPTION(int, snap::MeshBlockOptions, nghost)
      .ADD_OPTION(snap::IntegratorOptions, snap::MeshBlockOptions, intg)
      .ADD_OPTION(snap::HydroOptions, snap::MeshBlockOptions, hydro)
      .ADD_OPTION(snap::ScalarOptions, snap::MeshBlockOptions, scalar)
      .ADD_OPTION(std::vector<snap::BoundaryFlag>, snap::MeshBlockOptions,
                  bflags);

  torch::python::bind_module<snap::LogicalLocationImpl>(m, "LogicalLocation")
      .def(py::init<>())
      .def(py::init<int, int64_t, int64_t, int64_t>())
      .def("__repr__",
           [](const snap::LogicalLocationImpl &a) {
             return fmt::format(
                 "LogicalLocation(level = {}; lx1 = {:b}; lx2 = {:b}; lx3 = "
                 "{:b}",
                 a.level, a.lx1, a.lx2, a.lx3);
           })
      .def("__lt__", &snap::LogicalLocationImpl::lesser)
      .def("__gt__", &snap::LogicalLocationImpl::greater)
      .def("__eq__", &snap::LogicalLocationImpl::equal)
      .def_readwrite("lx1", &snap::LogicalLocationImpl::lx1)
      .def_readwrite("lx2", &snap::LogicalLocationImpl::lx2)
      .def_readwrite("lx3", &snap::LogicalLocationImpl::lx3)
      .def_readwrite("level", &snap::LogicalLocationImpl::level);

  torch::python::bind_module<snap::OctTreeNodeImpl>(m, "OctTreeNode")
      .def_readwrite("leaves", &snap::OctTreeNodeImpl::leaves)
      .def_readwrite("loc", &snap::OctTreeNodeImpl::loc)
      .def("nleaf", &snap::OctTreeNodeImpl::nleaf)
      .def("find_node", &snap::OctTreeNodeImpl::find_node);

  torch::python::bind_module<snap::OctTreeImpl>(m, "OctTree")
      .def(py::init<>())
      .def(py::init<snap::OctTreeOptions>())
      //.def("root", [](snap::OctTreeImpl &self) { return self.root.ptr(); })
      .def("root_level", &snap::OctTreeImpl::root_level)
      .def("forward", &snap::OctTreeImpl::forward);

  torch::python::bind_module<snap::MeshBlockImpl>(m, "MeshBlock")
      .def(py::init<>())
      .def(py::init<snap::MeshBlockOptions>())
      .def(py::init<snap::MeshBlockOptions, snap::LogicalLocation>())
      .def_readwrite("bfuncs", &snap::MeshBlockImpl::bfuncs)
      .def("nc1", &snap::MeshBlockImpl::nc1)
      .def("nc2", &snap::MeshBlockImpl::nc2)
      .def("nc3", &snap::MeshBlockImpl::nc3)
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
      .def("set_primitives", &snap::MeshBlockImpl::set_primitives,
           py::arg("hydro_w"), py::arg("scalar_w") = torch::nullopt)
      .def("max_root_time_step", &snap::MeshBlockImpl::max_root_time_step,
           py::arg("root_level") = 0, py::arg("solid") = torch::nullopt)
      .def("max_time_step", &snap::MeshBlockImpl::max_root_time_step,
           py::arg("root_level") = 0, py::arg("solid") = torch::nullopt)
      .def("forward", &snap::MeshBlockImpl::forward, py::arg("dt"),
           py::arg("stage"), py::arg("solid") = torch::nullopt)
      .def("set_uov",
           [](snap::MeshBlockImpl &self, std::string name, torch::Tensor val) {
             if (self.user_out_var.contains(name)) {
               self.user_out_var[name] = val;
             } else {
               self.user_out_var.insert(name, val);
             }
           })
      .def("var", [](snap::MeshBlockImpl &self, std::string name) {
        return GET_SHARED(name);
      });

  torch::python::bind_module<snap::MeshImpl>(m, "Mesh")
      .def(py::init<>(), R"(
          Construct a default Mesh object.
        )")
      .def(py::init<snap::MeshOptions>(), R"(
          Construct a new Mesh object with the given options.
        )")
      .def_readwrite("start_time", &snap::MeshImpl::start_time, R"(
          Returns the start time of the simulation.
        )")
      .def_readwrite("current_time", &snap::MeshImpl::current_time, R"(
          Returns the current time of the simulation.
        )")
      .def_readwrite("tlim", &snap::MeshImpl::tlim, R"(
          Returns the time limit of the simulation.
        )")
      .def("max_time_step", &snap::MeshImpl::max_time_step, R"(
          Returns the maximum time step of the simulation.
        )")
      .def("forward", &snap::MeshImpl::forward, R"(
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
