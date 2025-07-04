// pybind11
#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

// snap
#include <snap/mesh/meshblock.hpp>
#include <snap/output/output_formats.hpp>
#include <snap/output/output_formatter.hpp>

// python
#include "pyoptions.hpp"

namespace py = pybind11;

void bind_output(py::module &m) {
  py::class_<snap::OutputOptions>(m, "OutputOptions")
      .def(py::init<>())
      .def("__repr__",
           [](const snap::OutputOptions &a) {
             return fmt::format("OutputOptions{}", a);
           })
      .ADD_OPTION(int, snap::OutputOptions, fid)
      .ADD_OPTION(float, snap::OutputOptions, dt)
      .ADD_OPTION(int, snap::OutputOptions, dcycle)
      .ADD_OPTION(bool, snap::OutputOptions, output_slicex1)
      .ADD_OPTION(bool, snap::OutputOptions, output_slicex2)
      .ADD_OPTION(bool, snap::OutputOptions, output_slicex3)
      .ADD_OPTION(bool, snap::OutputOptions, output_sumx1)
      .ADD_OPTION(bool, snap::OutputOptions, output_sumx2)
      .ADD_OPTION(bool, snap::OutputOptions, output_sumx3)
      .ADD_OPTION(bool, snap::OutputOptions, include_ghost_zones)
      .ADD_OPTION(bool, snap::OutputOptions, cartesian_vector)
      .ADD_OPTION(float, snap::OutputOptions, x1_slice)
      .ADD_OPTION(float, snap::OutputOptions, x2_slice)
      .ADD_OPTION(float, snap::OutputOptions, x3_slice)
      .ADD_OPTION(std::string, snap::OutputOptions, block_name)
      .ADD_OPTION(std::string, snap::OutputOptions, file_basename)
      .ADD_OPTION(std::string, snap::OutputOptions, variable)
      .ADD_OPTION(std::string, snap::OutputOptions, file_type)
      .ADD_OPTION(std::string, snap::OutputOptions, data_format);

  py::class_<snap::OutputType>(m, "OutputType")
      .def(py::init<>())
      .def(py::init<snap::OutputOptions>())
      .def("__repr__",
           [](const snap::OutputType &a) {
             return fmt::format("OutputType(file_number = {}; next_time = {})",
                                a.file_number, a.next_time);
           })
      .def("increment_file_number",
           [](snap::OutputType &a) { return ++a.file_number; });

  py::class_<snap::NetcdfOutput, snap::OutputType>(m, "NetcdfOutput")
      .def(py::init<snap::OutputOptions>())
      .def("__repr__",
           [](const snap::NetcdfOutput &a) {
             return fmt::format(
                 "NetcdfOutput(file_number = {}; next_time = {})",
                 a.file_number, a.next_time);
           })
      .def(
          "write_output_file",
          [](snap::NetcdfOutput &self, py::object block_obj, float time,
             int wtflag) {
            py::object cpp_module = block_obj.attr("cpp_module");
            auto pmb = cpp_module.cast<std::shared_ptr<snap::MeshBlockImpl>>();
            snap::OctTreeOptions tree;
            self.write_output_file(pmb, time, tree, wtflag);
          },
          py::arg("block"), py::arg("time"), py::arg("wtflag") = 0)
      .def("combine_blocks", &snap::NetcdfOutput::combine_blocks);
}
