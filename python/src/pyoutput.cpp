// pybind11
#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

// fvm
#include <fvm/mesh/mesh.hpp>
#include <fvm/mesh/meshblock.hpp>
#include <fvm/output/output_formats.hpp>
#include <fvm/output/output_formatter.hpp>

// python
#include "pyoptions.hpp"

namespace py = pybind11;

void bind_output(py::module &m) {
  py::class_<canoe::OutputOptions>(m, "OutputOptions")
      .def(py::init<>())
      .def("__repr__",
           [](const canoe::OutputOptions &a) {
             return fmt::format("OutputOptions{}", a);
           })
      .ADD_OPTION(int, canoe::OutputOptions, fid)
      .ADD_OPTION(float, canoe::OutputOptions, dt)
      .ADD_OPTION(int, canoe::OutputOptions, dcycle)
      .ADD_OPTION(bool, canoe::OutputOptions, output_slicex1)
      .ADD_OPTION(bool, canoe::OutputOptions, output_slicex2)
      .ADD_OPTION(bool, canoe::OutputOptions, output_slicex3)
      .ADD_OPTION(bool, canoe::OutputOptions, output_sumx1)
      .ADD_OPTION(bool, canoe::OutputOptions, output_sumx2)
      .ADD_OPTION(bool, canoe::OutputOptions, output_sumx3)
      .ADD_OPTION(bool, canoe::OutputOptions, include_ghost_zones)
      .ADD_OPTION(bool, canoe::OutputOptions, cartesian_vector)
      .ADD_OPTION(float, canoe::OutputOptions, x1_slice)
      .ADD_OPTION(float, canoe::OutputOptions, x2_slice)
      .ADD_OPTION(float, canoe::OutputOptions, x3_slice)
      .ADD_OPTION(std::string, canoe::OutputOptions, block_name)
      .ADD_OPTION(std::string, canoe::OutputOptions, file_basename)
      .ADD_OPTION(std::string, canoe::OutputOptions, variable)
      .ADD_OPTION(std::string, canoe::OutputOptions, file_type)
      .ADD_OPTION(std::string, canoe::OutputOptions, data_format);

  py::class_<canoe::OutputType>(m, "OutputType")
      .def(py::init<>())
      .def(py::init<canoe::OutputOptions>())
      .def("__repr__",
           [](const canoe::OutputType &a) {
             return fmt::format("OutputType(file_number = {}; next_time = {})",
                                a.file_number, a.next_time);
           })
      .def("increment_file_number",
           [](canoe::OutputType &a) { return ++a.file_number; });

  py::class_<canoe::NetcdfOutput, canoe::OutputType>(m, "NetcdfOutput")
      .def(py::init<canoe::OutputOptions>())
      .def("__repr__",
           [](const canoe::NetcdfOutput &a) {
             return fmt::format(
                 "NetcdfOutput(file_number = {}; next_time = {})",
                 a.file_number, a.next_time);
           })
      .def(
          "write_output_file",
          [](canoe::NetcdfOutput &self, py::object block_obj, float time,
             canoe::OctTreeOptions const &tree, int wtflag) {
            py::object cpp_module = block_obj.attr("cpp_module");
            auto pmb = cpp_module.cast<std::shared_ptr<canoe::MeshBlockImpl>>();
            self.write_output_file(pmb, time, tree, wtflag);
          },
          py::arg("block"), py::arg("time"),
          py::arg("tree") = canoe::OctTreeOptions(), py::arg("wtflag") = 0)
      .def("combine_blocks", &canoe::NetcdfOutput::combine_blocks);
}
