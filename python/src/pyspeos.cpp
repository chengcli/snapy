// pybind11
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

// speos
#include <eos/speos.h>

namespace py = pybind11;

void bind_speos(py::module &m) {
  py::class_<spEosLeaf>(m, "SpeosLeaf")
      .def(py::init<>())
      .def_readwrite("type", &spEosLeaf::type)
      .def_property(
          "dens",
          [](const spEosLeaf &leaf) {
            return std::vector<double>(leaf.dens, leaf.dens + 2);
          },
          [](spEosLeaf &leaf, std::vector<double> v) {
            std::copy(v.begin(), v.end(), leaf.dens);
          })
      .def_property(
          "temp",
          [](const spEosLeaf &leaf) {
            return std::vector<double>(leaf.temp, leaf.temp + 2);
          },
          [](spEosLeaf &leaf, std::vector<double> v) {
            std::copy(v.begin(), v.end(), leaf.temp);
          })
      .def_property(
          "PA",
          [](const spEosLeaf &leaf) {
            return std::vector<double>(leaf.PA, leaf.PA + 16);
          },
          [](spEosLeaf &leaf, std::vector<double> v) {
            std::copy(v.begin(), v.end(), leaf.PA);
          })
      .def_property(
          "EA",
          [](const spEosLeaf &leaf) {
            return std::vector<double>(leaf.EA, leaf.EA + 16);
          },
          [](spEosLeaf &leaf, std::vector<double> v) {
            std::copy(v.begin(), v.end(), leaf.EA);
          })
      .def_readwrite("P_mx_err", &spEosLeaf::P_mx_err)
      .def_readwrite("E_mx_err", &spEosLeaf::E_mx_err)
      .def_readwrite("P_sd_err", &spEosLeaf::P_sd_err)
      .def_readwrite("E_sd_err", &spEosLeaf::E_sd_err)
      .def_readwrite("Cs_sd_err", &spEosLeaf::Cs_sd_err)
      .def_readwrite("Cs_mx_err", &spEosLeaf::Cs_mx_err)
      .def_readwrite("Cs_min", &spEosLeaf::Cs_min)
      .def_readwrite("Cs_max", &spEosLeaf::Cs_max)
      .def_readwrite("level", &spEosLeaf::level)
      .def_property(
          "child",
          [](const spEosLeaf &leaf) {
            return std::vector<std::vector<int>>{
                {leaf.child[0][0], leaf.child[0][1]},
                {leaf.child[1][0], leaf.child[1][1]}};
          },
          [](spEosLeaf &leaf, std::vector<std::vector<int>> v) {
            leaf.child[0][0] = v[0][0];
            leaf.child[0][1] = v[0][1];
            leaf.child[1][0] = v[1][0];
            leaf.child[1][1] = v[1][1];
          });

  py::class_<spEosHeader>(m, "SpeosHeader")
      .def(py::init<>())
      .def_readwrite("version", &spEosHeader::version)
      .def_readwrite("N0_dens", &spEosHeader::N0_dens)
      .def_readwrite("N0_temp", &spEosHeader::N0_temp)
      .def_readwrite("max_recursion_level", &spEosHeader::max_recursion_level)
      .def_readwrite("Nleaves", &spEosHeader::Nleaves)
      .def_readwrite("ref_dens", &spEosHeader::ref_dens)
      .def_readwrite("ref_temp", &spEosHeader::ref_temp)
      .def_property(
          "dens",
          [](const spEosHeader &hdr) {
            return std::vector<double>(hdr.dens, hdr.dens + 2);
          },
          [](spEosHeader &hdr, std::vector<double> v) {
            std::copy(v.begin(), v.end(), hdr.dens);
          })
      .def_property(
          "temp",
          [](const spEosHeader &hdr) {
            return std::vector<double>(hdr.temp, hdr.temp + 2);
          },
          [](spEosHeader &hdr, std::vector<double> v) {
            std::copy(v.begin(), v.end(), hdr.temp);
          })
      .def_property(
          "valid_dens",
          [](const spEosHeader &hdr) {
            return std::vector<double>(hdr.valid_dens, hdr.valid_dens + 2);
          },
          [](spEosHeader &hdr, std::vector<double> v) {
            std::copy(v.begin(), v.end(), hdr.valid_dens);
          })
      .def_property(
          "valid_temp",
          [](const spEosHeader &hdr) {
            return std::vector<double>(hdr.valid_temp, hdr.valid_temp + 2);
          },
          [](spEosHeader &hdr, std::vector<double> v) {
            std::copy(v.begin(), v.end(), hdr.valid_temp);
          })
      .def_property(
          "magic",
          [](const spEosHeader &hdr) { return std::string(hdr.magic); },
          [](spEosHeader &hdr, const std::string &v) {
            strncpy(hdr.magic, v.c_str(), 31);
            hdr.magic[31] = '\0';
          })
      .def_property(
          "eos_name",
          [](const spEosHeader &hdr) { return std::string(hdr.eos_name); },
          [](spEosHeader &hdr, const std::string &v) {
            strncpy(hdr.eos_name, v.c_str(), 31);
            hdr.eos_name[31] = '\0';
          })
      .def_property(
          "description",
          [](const spEosHeader &hdr) { return std::string(hdr.description); },
          [](spEosHeader &hdr, const std::string &v) {
            strncpy(hdr.description, v.c_str(), 255);
            hdr.description[255] = '\0';
          })
      .def_property(
          "date", [](const spEosHeader &hdr) { return std::string(hdr.date); },
          [](spEosHeader &hdr, const std::string &v) {
            strncpy(hdr.date, v.c_str(), 31);
            hdr.date[31] = '\0';
          });

  m.def("read_speos", [](const std::string &filename) {
    spEosHeader eos_output;
    spEosLeaf *eos_leaves = nullptr;
    read_speos(const_cast<char *>(filename.c_str()), &eos_output, &eos_leaves);
    return std::make_tuple(eos_output, eos_leaves);
  });

  m.def("dens_temp_speos",
        [](spEosHeader &speos_header, spEosLeaf *speos_leaves, double dens,
           double temp) {
          double press, energy, cs, dpdr, dpdt, dedr, dedt;
          dens_temp_speos(&speos_header, speos_leaves, dens, temp, &press,
                          &energy, &cs, &dpdr, &dpdt, &dedr, &dedt);
          return std::make_tuple(press, energy, cs, dpdr, dpdt, dedr, dedt);
        });

  m.def("dens_energy_speos",
        [](spEosHeader &speos_header, spEosLeaf *speos_leaves, double dens,
           double energy) {
          double press, temp, cs, dpdr, dpdt, dedr, dedt;
          dens_energy_speos(&speos_header, speos_leaves, dens, energy, &press,
                            &temp, &cs, &dpdr, &dpdt, &dedr, &dedt);
          return std::make_tuple(press, temp, cs, dpdr, dpdt, dedr, dedt);
        });

  m.def("dens_press_speos", [](spEosHeader &speos_header,
                               spEosLeaf *speos_leaves, double dens,
                               double press) {
    double temp, energy;
    dens_press_speos(&speos_header, speos_leaves, dens, press, &temp, &energy);
    return std::make_tuple(temp, energy);
  });
}
