// torch
#include <torch/extension.h>

// input
#include <input/parameter_input.hpp>

// fvm
#include <fvm/thermo/thermo_formatter.hpp>
#include <fvm/thermo/thermodynamics.hpp>

// python
#include "pyoptions.hpp"

namespace py = pybind11;

void bind_thermo(py::module &m) {
  py::class_<canoe::Nucleation>(m, "Nucleation")
      .def(py::init<>())
      .def(py::init<std::string const &, std::string const &,
                    std::map<std::string, double> const &>(),
           py::arg("equation"), py::arg("name"),
           py::arg("params") = std::map<std::string, double>())
      .def("__repr__",
           [](const canoe::Nucleation &a) {
             return fmt::format("Nucleation{}", a);
           })
      .ADD_OPTION(double, canoe::Nucleation, min_tem)
      .ADD_OPTION(double, canoe::Nucleation, max_tem)
      .ADD_OPTION(canoe::Reaction, canoe::Nucleation, reaction)
      .ADD_OPTION(canoe::func1_t, canoe::Nucleation, func)
      .ADD_OPTION(canoe::func1_t, canoe::Nucleation, logf_ddT);

  py::class_<canoe::CondensationOptions>(m, "CondensationOptions")
      .def(py::init<>())
      .def("__repr__",
           [](const canoe::CondensationOptions &a) {
             return fmt::format("CondensationOptions{}", a);
           })
      .ADD_OPTION(int, canoe::CondensationOptions, max_iter)
      .ADD_OPTION(std::string, canoe::CondensationOptions, dry_name)
      .ADD_OPTION(std::vector<canoe::Nucleation>, canoe::CondensationOptions,
                  react)
      .ADD_OPTION(std::vector<std::string>, canoe::CondensationOptions,
                  species);

  py::class_<canoe::ThermodynamicsOptions>(m, "ThermodynamicsOptions")
      .def(py::init<>())
      .def(py::init<canoe::ParameterInput>())
      .def("__repr__",
           [](const canoe::ThermodynamicsOptions &a) {
             return fmt::format("ThermodynamicsOptions{}", a);
           })
      .ADD_OPTION(double, canoe::ThermodynamicsOptions, Rd)
      .ADD_OPTION(double, canoe::ThermodynamicsOptions, gammad_ref)
      .ADD_OPTION(int, canoe::ThermodynamicsOptions, nvapor)
      .ADD_OPTION(int, canoe::ThermodynamicsOptions, ncloud)
      .ADD_OPTION(std::vector<std::string>, canoe::ThermodynamicsOptions,
                  species)
      .ADD_OPTION(std::vector<double>, canoe::ThermodynamicsOptions,
                  mu_ratio_m1)
      .ADD_OPTION(std::vector<double>, canoe::ThermodynamicsOptions,
                  cv_ratio_m1)
      .ADD_OPTION(std::vector<double>, canoe::ThermodynamicsOptions,
                  cp_ratio_m1)
      .ADD_OPTION(std::vector<double>, canoe::ThermodynamicsOptions, h0)
      .ADD_OPTION(int, canoe::ThermodynamicsOptions, max_iter)
      .ADD_OPTION(double, canoe::ThermodynamicsOptions, ftol)
      .ADD_OPTION(double, canoe::ThermodynamicsOptions, boost)
      .ADD_OPTION(canoe::CondensationOptions, canoe::ThermodynamicsOptions,
                  cond);

  ADD_CANOE_MODULE(Condensation, CondensationOptions)
      .def("species_index", &canoe::CondensationImpl::species_index)
      .def("forward", &canoe::CondensationImpl::forward);

  ADD_CANOE_MODULE(Thermodynamics, ThermodynamicsOptions)
      .def_readonly("options", &canoe::ThermodynamicsImpl::options)
      .def("get_gammad", &canoe::ThermodynamicsImpl::get_gammad, py::arg("var"),
           py::arg("type") = 0)
      .def("f_eps", &canoe::ThermodynamicsImpl::f_eps)
      .def("f_sig", &canoe::ThermodynamicsImpl::f_sig)
      .def("f_psi", &canoe::ThermodynamicsImpl::f_psi)
      .def("get_mu", &canoe::ThermodynamicsImpl::get_mu)
      .def("get_cv", &canoe::ThermodynamicsImpl::get_cv_ref)
      .def("get_cp", &canoe::ThermodynamicsImpl::get_cp_ref)
      .def("get_temp", &canoe::ThermodynamicsImpl::get_temp)
      .def("get_theta", &canoe::ThermodynamicsImpl::get_theta_ref)
      .def("species_index", &canoe::ThermodynamicsImpl::species_index)
      .def("forward", &canoe::ThermodynamicsImpl::forward);
}
