// torch
#include <torch/extension.h>

// snap
#include <snap/input/parameter_input.hpp>
#include <snap/thermo/thermo_formatter.hpp>
#include <snap/thermo/thermodynamics.hpp>

// python
#include "pyoptions.hpp"

namespace py = pybind11;

void bind_thermo(py::module &m) {
  py::class_<snap::Nucleation>(m, "Nucleation")
      .def(py::init<>())
      .def(py::init<std::string const &, std::string const &,
                    std::map<std::string, double> const &>(),
           py::arg("equation"), py::arg("name"),
           py::arg("params") = std::map<std::string, double>())
      .def("__repr__",
           [](const snap::Nucleation &a) {
             return fmt::format("Nucleation{}", a);
           })
      .ADD_OPTION(double, snap::Nucleation, min_tem)
      .ADD_OPTION(double, snap::Nucleation, max_tem)
      .ADD_OPTION(snap::Reaction, snap::Nucleation, reaction)
      .ADD_OPTION(snap::func1_t, snap::Nucleation, func)
      .ADD_OPTION(snap::func1_t, snap::Nucleation, logf_ddT);

  py::class_<snap::CondensationOptions>(m, "CondensationOptions")
      .def(py::init<>())
      .def("__repr__",
           [](const snap::CondensationOptions &a) {
             return fmt::format("CondensationOptions{}", a);
           })
      .ADD_OPTION(int, snap::CondensationOptions, max_iter)
      .ADD_OPTION(std::string, snap::CondensationOptions, dry_name)
      .ADD_OPTION(std::vector<snap::Nucleation>, snap::CondensationOptions,
                  react)
      .ADD_OPTION(std::vector<std::string>, snap::CondensationOptions, species);

  py::class_<snap::ThermodynamicsOptions>(m, "ThermodynamicsOptions")
      .def(py::init<>())
      .def(py::init<snap::ParameterInput>())
      .def("__repr__",
           [](const snap::ThermodynamicsOptions &a) {
             return fmt::format("ThermodynamicsOptions{}", a);
           })
      .ADD_OPTION(double, snap::ThermodynamicsOptions, Rd)
      .ADD_OPTION(double, snap::ThermodynamicsOptions, gammad_ref)
      .ADD_OPTION(int, snap::ThermodynamicsOptions, nvapor)
      .ADD_OPTION(int, snap::ThermodynamicsOptions, ncloud)
      .ADD_OPTION(std::vector<std::string>, snap::ThermodynamicsOptions,
                  species)
      .ADD_OPTION(std::vector<double>, snap::ThermodynamicsOptions, mu_ratio_m1)
      .ADD_OPTION(std::vector<double>, snap::ThermodynamicsOptions, cv_ratio_m1)
      .ADD_OPTION(std::vector<double>, snap::ThermodynamicsOptions, cp_ratio_m1)
      .ADD_OPTION(std::vector<double>, snap::ThermodynamicsOptions, h0)
      .ADD_OPTION(int, snap::ThermodynamicsOptions, max_iter)
      .ADD_OPTION(double, snap::ThermodynamicsOptions, ftol)
      .ADD_OPTION(double, snap::ThermodynamicsOptions, boost)
      .ADD_OPTION(snap::CondensationOptions, snap::ThermodynamicsOptions, cond);

  ADD_SNAP_MODULE(Condensation, CondensationOptions)
      .def("species_index", &snap::CondensationImpl::species_index)
      .def("forward", &snap::CondensationImpl::forward);

  ADD_SNAP_MODULE(Thermodynamics, ThermodynamicsOptions)
      .def_readonly("options", &snap::ThermodynamicsImpl::options)
      .def("get_gammad", &snap::ThermodynamicsImpl::get_gammad, py::arg("var"),
           py::arg("type") = 0)
      .def("f_eps", &snap::ThermodynamicsImpl::f_eps)
      .def("f_sig", &snap::ThermodynamicsImpl::f_sig)
      .def("f_psi", &snap::ThermodynamicsImpl::f_psi)
      .def("get_mu", &snap::ThermodynamicsImpl::get_mu)
      .def("get_cv", &snap::ThermodynamicsImpl::get_cv_ref)
      .def("get_cp", &snap::ThermodynamicsImpl::get_cp_ref)
      .def("get_temp", &snap::ThermodynamicsImpl::get_temp)
      .def("get_theta", &snap::ThermodynamicsImpl::get_theta_ref)
      .def("species_index", &snap::ThermodynamicsImpl::species_index)
      .def("forward", &snap::ThermodynamicsImpl::forward);
}
