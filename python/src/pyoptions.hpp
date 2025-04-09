
#define ADD_OPTION(T, st_name, op_name)                             \
  def(#op_name, (T const &(st_name::*)() const) & st_name::op_name) \
      .def(#op_name, (st_name & (st_name::*)(const T &)) & st_name::op_name)

#define ADD_SNAP_MODULE(m_name, op_name)                       \
  torch::python::bind_module<snap::m_name##Impl>(m, #m_name)   \
      .def(py::init<>(), R"(Construct a new default module.)") \
      .def(py::init<snap::op_name>(), R"(                     \
        Construct a new module with options)")                 \
      .def("__repr__",                                         \
           [](const snap::m_name##Impl &a) {                   \
             return fmt::format(#m_name "{}", a.options);      \
           })                                                  \
      .def("module",                                           \
           [](snap::m_name##Impl &self, std::string name) {    \
             return self.named_modules()[name];                \
           })                                                  \
      .def("buffer",                                           \
           [](snap::m_name##Impl &self, std::string name) {    \
             return self.named_buffers()[name];                \
           })                                                  \
      .def("forward", &snap::m_name##Impl::forward)
