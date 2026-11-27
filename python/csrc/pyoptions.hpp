#include <fmt/format.h>

// torch
#include <torch/extension.h>

// snap
#include <snap/utils/step_dt.hpp>

// V3 D: a stage's dt (StepDt) is a Python float, as before
namespace pybind11::detail {
template <>
struct type_caster<snap::StepDt> {
  PYBIND11_TYPE_CASTER(snap::StepDt, const_name("float"));
  bool load(handle src, bool convert) {
    make_caster<double> d;
    if (!d.load(src, convert)) return false;
    value = snap::StepDt(cast_op<double>(d));
    return true;
  }
  static handle cast(snap::StepDt const& dt, return_value_policy, handle) {
    return PyFloat_FromDouble(dt.value());
  }
};
}  // namespace pybind11::detail

namespace snap_python {

template <typename T>
constexpr bool printable_option =
    std::is_arithmetic_v<T> || std::is_same_v<T, std::string>;

template <typename T>
constexpr bool printable_option<std::vector<T>> = printable_option<T>;

// Register printable option names alongside the Python getters. Reading only
// those getters avoids invoking arbitrary zero-argument methods during print.
template <typename Options, typename T>
auto option_getter(char const* name, T const& (Options::*getter)() const) {
  if constexpr (printable_option<T>) {
    auto cls = pybind11::type::of<Options>();
    if (!pybind11::hasattr(cls, "_option_fields"))
      cls.attr("_option_fields") = pybind11::list();
    cls.attr("_option_fields").attr("append")(name);
  }
  return getter;
}

inline std::string options_repr(pybind11::object self) {
  auto cls = pybind11::type::of(self);
  pybind11::list fields;
  for (auto name : cls.attr("_option_fields")) {
    auto value = self.attr(name)();
    fields.append(pybind11::str("{}={!r}").attr("format")(name, value));
  }
  return pybind11::str("{}({})")
      .attr("format")(cls.attr("__name__"),
                      pybind11::str(", ").attr("join")(fields))
      .cast<std::string>();
}

template <typename Module>
std::string module_repr(Module const& module) {
  auto cls = pybind11::type::of<Module>();
  return pybind11::str("{}(options={!r})")
      .attr("format")(cls.attr("__name__"), pybind11::cast(module.options))
      .template cast<std::string>();
}

}  // namespace snap_python

#define ADD_OPTION(T, st_name, op_name)                                    \
  def(#op_name,                                                            \
      snap_python::option_getter<st_name, T>(#op_name, &st_name::op_name)) \
      .def(#op_name, (st_name & (st_name::*)(const T&)) & st_name::op_name)

#define ADD_SNAP_MODULE(m_name, op_name)                              \
  torch::python::bind_module<snap::m_name##Impl>(m, #m_name)          \
      .def(py::init<>(), R"(Construct a new default module.)")        \
      .def_readonly("options", &snap::m_name##Impl::options)          \
      .def("__repr__", &snap_python::module_repr<snap::m_name##Impl>, \
           py::prepend())                                             \
      .def("__str__", &snap_python::module_repr<snap::m_name##Impl>,  \
           py::prepend())                                             \
      .def("module",                                                  \
           [](snap::m_name##Impl& self, std::string name) {           \
             return self.named_modules()[name];                       \
           })                                                         \
      .def("buffer",                                                  \
           [](snap::m_name##Impl& self, std::string name) {           \
             return self.named_buffers()[name];                       \
           })                                                         \
      .def("forward", &snap::m_name##Impl::forward,                   \
           py::call_guard<py::gil_scoped_release>())

// A stepped module (forward(vars, dt, stage)) binds its own forward with the
// GIL released; torch::python::bind_module's automatic forward/__call__ holds
// the GIL and would shadow it (V3 S-K4: the chemistry step's autograd use
// inside forward() needs the GIL released)
#define ADD_SNAP_MODULE_STEPPED(m_name, op_name)                      \
  torch::python::bind_module<snap::m_name##Impl, true>(m, #m_name)    \
      .def(py::init<>(), R"(Construct a new default module.)")        \
      .def_readonly("options", &snap::m_name##Impl::options)          \
      .def("__repr__", &snap_python::module_repr<snap::m_name##Impl>, \
           py::prepend())                                             \
      .def("__str__", &snap_python::module_repr<snap::m_name##Impl>,  \
           py::prepend())                                             \
      .def("module",                                                  \
           [](snap::m_name##Impl& self, std::string name) {           \
             return self.named_modules()[name];                       \
           })                                                         \
      .def("buffer",                                                  \
           [](snap::m_name##Impl& self, std::string name) {           \
             return self.named_buffers()[name];                       \
           })                                                         \
      .def("forward", &snap::m_name##Impl::forward,                   \
           py::call_guard<py::gil_scoped_release>())
