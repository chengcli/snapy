// pybind11
#include <pybind11/functional.h>
#include <pybind11/stl.h>

// fmt
#include <fmt/format.h>

// torch
#include <torch/extension.h>
#include <torch/nn/modules/container/any.h>

// snap
#include <snap/layout/cubed_layout.hpp>
#include <snap/layout/cubed_sphere_layout.hpp>
#include <snap/layout/distribute_info.hpp>
#include <snap/layout/exchange.hpp>
#include <snap/layout/slab_layout.hpp>
#include <snap/mesh/meshblock.hpp>

// python
#include "pyoptions.hpp"

namespace py = pybind11;

void bind_layout(py::module& m) {
  auto pyDistributeInfo = py::class_<snap::DistributeInfo>(m, "DistributeInfo");

  pyDistributeInfo.def(py::init<>())
      .def("__repr__",
           [](const snap::DistributeInfo& a) {
             std::stringstream ss;
             a.report(ss);
             return fmt::format("DistributeInfo(\n{})", ss.str());
           })
      .ADD_OPTION(int, snap::DistributeInfo, face)
      .ADD_OPTION(int, snap::DistributeInfo, level)
      .ADD_OPTION(int, snap::DistributeInfo, gid)
      .ADD_OPTION(int, snap::DistributeInfo, lx1)
      .ADD_OPTION(int, snap::DistributeInfo, lx2)
      .ADD_OPTION(int, snap::DistributeInfo, lx3)
      .ADD_OPTION(int, snap::DistributeInfo, nb1)
      .ADD_OPTION(int, snap::DistributeInfo, nb2)
      .ADD_OPTION(int, snap::DistributeInfo, nb3);

  auto pySlabLayout = py::class_<snap::SlabLayout>(m, "SlabLayout");

  pySlabLayout
      .def(py::init<int, int, bool, bool>(), py::arg("px"), py::arg("py"),
           py::arg("periodic_x") = false, py::arg("periodic_y") = false)
      .def("__repr__",
           [](const snap::SlabLayout& a) {
             std::stringstream ss;
             a.report(ss);
             return fmt::format("SlabLayout(\n{})", ss.str());
           })
      .def("get_procs", &snap::SlabLayout::get_procs)
      .def("rank_of", &snap::SlabLayout::rank_of, py::arg("rx"), py::arg("ry"))
      .def("loc_of", &snap::SlabLayout::loc_of, py::arg("rank"))
      .def(
          "neighbor_rank",
          [](const snap::SlabLayout& self, int rx, int ry, int dx, int dy,
             int dz) { return self.neighbor_rank(rx, ry, dx, dy); },
          py::arg("rx"), py::arg("ry"), py::arg("dx"), py::arg("dy"),
          py::arg("dz") = 0);

  auto pyCubedLayout = py::class_<snap::CubedLayout>(m, "CubedLayout");

  pyCubedLayout
      .def(py::init<int, int, int, bool, bool, bool>(), py::arg("px"),
           py::arg("py"), py::arg("pz"), py::arg("periodic_x") = false,
           py::arg("periodic_y") = false, py::arg("periodic_z") = false)
      .def("__repr__",
           [](const snap::CubedLayout& a) {
             std::stringstream ss;
             a.report(ss);
             return fmt::format("CubedLayout(\n{})", ss.str());
           })
      .def("get_procs", &snap::CubedLayout::get_procs)
      .def("rank_of", &snap::CubedLayout::rank_of, py::arg("rx"), py::arg("ry"),
           py::arg("rz"))
      .def("loc_of", &snap::CubedLayout::loc_of, py::arg("rank"))
      .def("neighbor_rank", &snap::CubedLayout::neighbor_rank, py::arg("rx"),
           py::arg("ry"), py::arg("rz"), py::arg("dx"), py::arg("dy"),
           py::arg("dz"));

  auto pyCubedSphereLayout =
      py::class_<snap::CubedSphereLayout>(m, "CubedSphereLayout");

  pyCubedSphereLayout.def(py::init<int>(), py::arg("pxy"))
      .def("__repr__",
           [](const snap::CubedSphereLayout& a) {
             std::stringstream ss;
             a.report(ss);
             return fmt::format("CubedSphereLayout(\n{})", ss.str());
           })
      .def("get_procs", &snap::CubedSphereLayout::get_procs)
      .def("rank_of", &snap::CubedSphereLayout::rank_of, py::arg("face"),
           py::arg("rx"), py::arg("ry"))
      .def("loc_of", &snap::CubedSphereLayout::loc_of, py::arg("rank"))
      .def(
          "neighbor_rank",
          [](const snap::CubedSphereLayout& self, int face, int rx, int ry,
             int dx, int dy,
             int dz) { return self.neighbor_rank(face, rx, ry, dx, dy); },
          py::arg("face"), py::arg("rx"), py::arg("ry"), py::arg("dx"),
          py::arg("dy"), py::arg("dz") = 0);

  // Exchange functions
  m.def("get_buffer_id", &snap::get_buffer_id, py::arg("dx"), py::arg("dy"),
        py::arg("dz") = 0, "Calculate buffer ID from directional offsets");

  m.def(
      "init_buffers_2d",
      [](snap::MeshBlock const& block, torch::Tensor const& hydro_u) {
        std::vector<torch::Tensor> send_bufs, recv_bufs;
        snap::init_buffers_2d(block.ptr().get(), hydro_u, send_bufs, recv_bufs);
        return py::make_tuple(send_bufs, recv_bufs);
      },
      py::arg("block"), py::arg("hydro_u"),
      "Initialize send and receive buffers for 2D domain decomposition");

  m.def(
      "serialize_2d",
      [](snap::MeshBlock const& block, torch::Tensor& hydro_u,
         std::vector<torch::Tensor>& send_bufs) {
        snap::serialize_2d(block.ptr().get(), hydro_u, send_bufs);
      },
      py::arg("block"), py::arg("hydro_u"), py::arg("send_bufs"),
      "Serialize mesh data into send buffers");

  m.def(
      "deserialize_2d",
      [](snap::MeshBlock const& block, torch::Tensor& hydro_u,
         std::vector<torch::Tensor> const& recv_bufs) {
        snap::deserialize_2d(block.ptr().get(), hydro_u, recv_bufs);
      },
      py::arg("block"), py::arg("hydro_u"), py::arg("recv_bufs"),
      "Deserialize received data into mesh ghost zones");

  m.def(
      "init_dist",
      [](std::string const& backend, std::string const& init_method,
         std::string const& layout_type, int px, int py, int pz,
         bool periodic_x1, bool periodic_x2, bool periodic_x3,
         std::string const& device_type, int local_rank) {
        auto result = snap::init_dist(backend, init_method, layout_type, px, py,
                                      pz, periodic_x1, periodic_x2, periodic_x3,
                                      device_type, local_rank);
        // Return tuple: (layout, ranks, device, info, layout_type)
        // Note: layout is stored as void* internally, so we can't easily expose
        // it Instead, return the components that Python needs
        return py::make_tuple(result.ranks, result.device, result.info,
                              result.layout_type);
      },
      py::arg("backend"), py::arg("init_method"), py::arg("layout_type"),
      py::arg("px"), py::arg("py"), py::arg("pz"),
      py::arg("periodic_x1") = false, py::arg("periodic_x2") = false,
      py::arg("periodic_x3") = false, py::arg("device_type") = "cpu",
      py::arg("local_rank") = -1,
      "Initialize distributed computing environment");

  m.def(
      "slab_exchange",
      [](snap::MeshBlock const& block, torch::Tensor& hydro_u,
         std::vector<int> const& ranks, std::vector<torch::Tensor>& send_bufs,
         std::vector<torch::Tensor>& recv_bufs) {
        snap::slab_exchange(block.ptr().get(), hydro_u, ranks, send_bufs,
                            recv_bufs);
      },
      py::arg("block"), py::arg("hydro_u"), py::arg("ranks"),
      py::arg("send_bufs"), py::arg("recv_bufs"),
      "Perform ghost zone exchange for slab layout");
}
