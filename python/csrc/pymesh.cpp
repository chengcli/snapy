// pybind11
#include <pybind11/functional.h>
#include <pybind11/stl.h>

// torch
#include <torch/extension.h>

// snap
#include <snap/mesh/mesh.hpp>
#include <snap/mesh/meshblock.hpp>
#include <snap/output/output_formats.hpp>

// python
#include "pyoptions.hpp"

namespace py = pybind11;

void bind_mesh(py::module& m) {
  auto pyMeshBlockOptions =
      py::class_<snap::MeshBlockOptionsImpl, snap::MeshBlockOptions>(
          m, "MeshBlockOptions");
  auto pyMeshOptions =
      py::class_<snap::MeshOptionsImpl, snap::MeshOptions>(m, "MeshOptions");

  pyMeshBlockOptions.def(py::init<>())
      .def("__repr__",
           [](const snap::MeshBlockOptions& a) {
             std::stringstream ss;
             a->report(ss);
             return fmt::format("MeshBlockOptions(\n{})", ss.str());
           })
      .def_static("from_yaml", &snap::MeshBlockOptionsImpl::from_yaml,
                  py::arg("filename"), py::arg("verbose") = false)
      .def("device_str", &snap::MeshBlockOptionsImpl::device_str)
      .def(
          "set_bfunc",
          [&](snap::MeshBlockOptions& self, int dx3, int dx2, int dx1,
              py::object func_obj) {
            bcfunc_t func;
            if (func_obj.is_none()) {
              func = nullptr;
            } else {
              py::function f = py::cast<py::function>(func_obj);
              func = [f](torch::Tensor const& var, int id,
                         snap::BoundaryFuncOptions op) {
                py::gil_scoped_acquire gil;
                f(var, id, op);
              };
            }

            if (self->bfuncs().empty()) {
              throw std::runtime_error(
                  "Cannot set boundary function when bfuncs is empty.");
            } else if (self->bfuncs().size() == 2) {
              if (dx3 != 0 || dx2 != 0) {
                throw std::runtime_error(
                    "Only dx1 can be non-zero when bfuncs has size 2.");
              }
            } else if (self->bfuncs().size() == 4) {
              if (dx3 != 0) {
                throw std::runtime_error(
                    "Only dx1 and dx2 can be non-zero when bfuncs has size 4.");
              }
            } else if (self->bfuncs().size() != 6) {
              throw std::runtime_error(
                  "bfuncs must have size 2, 4, or 6 to set boundary "
                  "functions.");
            }

            if (dx3 == 0 && dx2 == 0 && dx1 == -1) {
              self->bfuncs()[0] = func;
            } else if (dx3 == 0 && dx2 == 0 && dx1 == 1) {
              self->bfuncs()[1] = func;
            } else if (dx3 == 0 && dx2 == -1 && dx1 == 0) {
              self->bfuncs()[2] = func;
            } else if (dx3 == 0 && dx2 == 1 && dx1 == 0) {
              self->bfuncs()[3] = func;
            } else if (dx3 == -1 && dx2 == 0 && dx1 == 0) {
              self->bfuncs()[4] = func;
            } else if (dx3 == 1 && dx2 == 0 && dx1 == 0) {
              self->bfuncs()[5] = func;
            }
          },
          py::arg("dx3"), py::arg("dx2"), py::arg("dx1"), py::arg("func"))
      .ADD_OPTION(bool, snap::MeshBlockOptionsImpl, verbose)
      .ADD_OPTION(std::string, snap::MeshBlockOptionsImpl, basename)
      .ADD_OPTION(std::string, snap::MeshBlockOptionsImpl, output_dir)
      .ADD_OPTION(std::vector<snap::OutputOptions>, snap::MeshBlockOptionsImpl,
                  outputs)
      .ADD_OPTION(harp::IntegratorOptions, snap::MeshBlockOptionsImpl, intg)
      .ADD_OPTION(snap::CoordinateOptions, snap::MeshBlockOptionsImpl, coord)
      .ADD_OPTION(snap::HydroOptions, snap::MeshBlockOptionsImpl, hydro)
      .ADD_OPTION(snap::ScalarOptions, snap::MeshBlockOptionsImpl, scalar)
      .ADD_OPTION(snap::InternalBoundaryOptions, snap::MeshBlockOptionsImpl, ib)
      .ADD_OPTION(std::vector<bcfunc_t>, snap::MeshBlockOptionsImpl, bfuncs)
      .ADD_OPTION(snap::LayoutOptions, snap::MeshBlockOptionsImpl, layout);

  pyMeshOptions.def(py::init<>())
      .def("__repr__",
           [](const snap::MeshOptions& a) {
             return fmt::format("MeshOptions(blocks_per_process={})",
                                a->blocks_per_process());
           })
      .def_static("from_yaml", &snap::MeshOptionsImpl::from_yaml,
                  py::arg("filename"), py::arg("verbose") = false)
      .def("device_str", &snap::MeshOptionsImpl::device_str)
      .def(
          "set_local_horizontal_cells",
          [](snap::MeshOptions& self, int nx2, int nx3) {
            if (nx2 <= 0 || nx3 <= 0) {
              throw std::invalid_argument(
                  "horizontal cell counts must be positive");
            }
            auto block = self->block();
            if (block == nullptr || block->coord() == nullptr ||
                block->layout() == nullptr) {
              throw std::runtime_error(
                  "MeshOptions requires coordinate and layout options");
            }
            auto coord = block->coord();
            auto layout = block->layout();
            coord->nx2(nx2);
            coord->nx3(nx3);
            coord->global_nx2(nx2 * layout->px());
            coord->global_nx3(nx3 * layout->py());
          },
          py::arg("nx2"), py::arg("nx3"))
      .ADD_OPTION(snap::MeshBlockOptions, snap::MeshOptionsImpl, block)
      .ADD_OPTION(int, snap::MeshOptionsImpl, blocks_per_process);

  ADD_SNAP_MODULE(MeshBlock, MeshBlockOptions)
      .def(py::init<snap::MeshBlockOptions>(), py::arg("options"))
      .def("cycle", [](snap::MeshBlockImpl& self) { return self.cycle; })
      .def("inc_cycle",
           [](snap::MeshBlockImpl& self) {
             auto v = self.cycle;
             self.cycle++;
             return v;
           })
      .def("set_user_output_func",
           [&](snap::MeshBlockImpl& self, py::object func_ojb) {
             py::function f = py::cast<py::function>(func_ojb);
             self.user_output_callback =
                 [f](std::map<std::string, torch::Tensor> const& vars) {
                   py::gil_scoped_acquire gil;
                   return f(vars).cast<std::map<std::string, torch::Tensor>>();
                 };
           })
      .def("set_user_forcing_func",
           [&](snap::MeshBlockImpl& self, py::object func_obj) {
             py::function f = py::cast<py::function>(func_obj);
             self.user_forcing_callback =
                 [f](std::map<std::string, torch::Tensor> const& vars,
                     double dt, int stage) {
                   py::gil_scoped_acquire gil;
                   return f(vars, dt, stage)
                       .cast<std::map<std::string, torch::Tensor>>();
                 };
           })
      .def("max_time_step", &snap::MeshBlockImpl::max_time_step)
      .def("make_outputs", &snap::MeshBlockImpl::make_outputs, py::arg("vars"),
           py::arg("time"), py::arg("final_write") = false)
      .def("forward", &snap::MeshBlockImpl::forward, py::arg("vars"),
           py::arg("dt"), py::arg("stage"),
           py::call_guard<py::gil_scoped_release>())
      .def(
          "part",
          [](snap::MeshBlockImpl& self, std::tuple<int, int, int> offset,
             bool exterior, int extend_x1, int extend_x2, int extend_x3) {
            snap::PartOptions opts;
            opts.exterior(exterior);
            opts.extend_x1(extend_x1);
            opts.extend_x2(extend_x2);
            opts.extend_x3(extend_x3);

            auto result = self.part(offset, opts);
            py::tuple index_spec(result.size());
            for (size_t i = 0; i < result.size(); ++i) {
              auto s = result[i].slice();
              index_spec[i] =
                  py::slice(s.start().expect_int(), s.stop().expect_int(),
                            s.step().expect_int());
            }
            return index_spec;
          },
          py::arg("offset"), py::arg("exterior") = true,
          py::arg("extend_x1") = 0, py::arg("extend_x2") = 0,
          py::arg("extend_x3") = 0)
      .def(
          "initialize",
          [](snap::MeshBlockImpl& self, snap::Variables& vars) {
            self.initialize(vars);
            return std::make_pair(vars, 0.);
          },
          py::arg("vars"))
      .def(
          "initialize_from_restart",
          [](snap::MeshBlockImpl& self, std::string restart_file) {
            snap::Variables vars;
            double time = self.initialize(vars, restart_file.c_str());
            return std::make_pair(vars, time);
          },
          py::arg("restart_file"))
      .def("initialize_local", &snap::MeshBlockImpl::initialize_local,
           py::arg("vars"))
      .def("initialize_under_mesh", &snap::MeshBlockImpl::initialize_under_mesh,
           py::arg("vars"))
      .def("finalize_initialization",
           &snap::MeshBlockImpl::finalize_initialization, py::arg("vars"))
      .def("local_max_time_step", &snap::MeshBlockImpl::local_max_time_step,
           py::arg("vars"))
      .def("advance_local", &snap::MeshBlockImpl::advance_local,
           py::arg("vars"), py::arg("dt"), py::arg("stage"),
           py::call_guard<py::gil_scoped_release>())
      .def("exchange_ghost_zones", &snap::MeshBlockImpl::exchange_ghost_zones,
           py::arg("vars"))
      .def("get_layout", &snap::MeshBlockImpl::get_layout)
      .def("print_cycle_info", &snap::MeshBlockImpl::print_cycle_info)
      .def("finalize", &snap::MeshBlockImpl::finalize)
      .def("check_redo", &snap::MeshBlockImpl::check_redo)
      .def("get_outputs",
           [](snap::MeshBlockImpl& self) { return self.output_types; })
      .def(
          "apply_hydro_bc",
          [](snap::MeshBlockImpl& self, torch::Tensor var, int type) {
            snap::BoundaryFuncOptions bops;
            bops.nghost(self.options->coord()->nghost());
            if (type != snap::kPrimitive && type != snap::kConserved) {
              throw std::runtime_error("Invalid type for apply_hydro_bc.");
            }
            bops.type(type);
            for (size_t i = 0; i < self.options->bfuncs().size(); ++i) {
              if (self.options->bfuncs()[i] == nullptr) continue;
              self.options->bfuncs()[i](var, 3 - i / 2, bops);
            }
          },
          py::arg("var"), py::arg("type") = (int)snap::kConserved);

  ADD_SNAP_MODULE(Mesh, MeshOptions)
      .def(py::init<snap::MeshOptions>(), py::arg("options"))
      .def_property_readonly("blocks",
                             [](const snap::MeshImpl& self) {
                               py::list out;
                               for (auto& b : self.blocks) {
                                 std::shared_ptr<snap::MeshBlockImpl> p =
                                     b.ptr();
                                 out.append(py::cast(std::move(p)));
                               }
                               return out;
                             })
      .def(
          "initialize",
          [](snap::MeshImpl& self, snap::MeshVariables& vars) {
            self.initialize(vars);
            return std::make_pair(vars, 0.);
          },
          py::arg("vars"))
      .def(
          "initialize_from_restart",
          [](snap::MeshImpl& self, std::string restart_file) {
            snap::MeshVariables vars(self.blocks.size());
            double time = self.initialize(vars, restart_file.c_str());
            return std::make_pair(vars, time);
          },
          py::arg("restart_file"))
      .def("max_time_step", &snap::MeshImpl::max_time_step, py::arg("vars"))
      .def("exchange_ghost_zones", &snap::MeshImpl::exchange_ghost_zones,
           py::arg("vars"), py::arg("type") = (int)snap::kConserved,
           py::call_guard<py::gil_scoped_release>())
      .def("make_outputs", &snap::MeshImpl::make_outputs, py::arg("vars"),
           py::arg("current_time"), py::arg("final_write") = false)
      .def("print_cycle_info", &snap::MeshImpl::print_cycle_info,
           py::arg("vars"), py::arg("time"), py::arg("dt"))
      .def("check_redo", &snap::MeshImpl::check_redo, py::arg("vars"))
      .def("set_cycle", &snap::MeshImpl::set_cycle, py::arg("cycle"))
      .def("finalize", &snap::MeshImpl::finalize, py::arg("vars"),
           py::arg("time"));
}
