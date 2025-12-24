#pragma once

// C/C++
#include <memory>

// yaml
#include <yaml-cpp/yaml.h>

// torch
#include <torch/torch.h>

// base
#include <configure.h>

// kintera
#include <kintera/utils/format.hpp>

// snap
#include <snap/coord/coordinate.hpp>
#include <snap/interface/athena_arrays.hpp>

// arg
#include <snap/add_arg.h>

namespace snap {

class MeshBlockImpl;
using Variables = std::map<std::string, torch::Tensor>;

//! \brief  container for parameters read from `<output>` block in the input
struct OutputOptionsImpl {
  static std::shared_ptr<OutputOptionsImpl> create() {
    return std::make_shared<OutputOptionsImpl>();
  }
  static std::shared_ptr<OutputOptionsImpl> from_yaml(YAML::Node const &node,
                                                      int fid = 0);
  std::string file_id() const { return "out" + std::to_string(fid()); }

  void report(std::ostream &os) const {
    os << "-- output options --\n";
    os << "* fid = " << fid() << "\n"
       << "* dt = " << dt() << "\n"
       << "* output_slicex1 = " << output_slicex1() << "\n"
       << "* output_slicex2 = " << output_slicex2() << "\n"
       << "* output_slicex3 = " << output_slicex3() << "\n"
       << "* output_sumx1 = " << output_sumx1() << "\n"
       << "* output_sumx2 = " << output_sumx2() << "\n"
       << "* output_sumx3 = " << output_sumx3() << "\n"
       << "* include_ghost_zones = " << include_ghost_zones() << "\n"
       << "* cartesian_vector = " << cartesian_vector() << "\n"
       << "* x1_slice = " << x1_slice() << "\n"
       << "* x2_slice = " << x2_slice() << "\n"
       << "* x3_slice = " << x3_slice() << "\n"
       << "* file_type = " << file_type() << "\n"
       << "* data_format = " << data_format() << "\n"
       << "* variables = " << fmt::format("{}", variables()) << "\n"
       << "* combine = " << combine() << "\n"
       << "* verbose = " << verbose() << "\n";
  }

  ADD_ARG(int, fid) = 0;
  ADD_ARG(double, dt) = 0.;

  ADD_ARG(bool, output_slicex1) = false;
  ADD_ARG(bool, output_slicex2) = false;
  ADD_ARG(bool, output_slicex3) = false;

  ADD_ARG(bool, output_sumx1) = false;
  ADD_ARG(bool, output_sumx2) = false;
  ADD_ARG(bool, output_sumx3) = false;

  ADD_ARG(bool, include_ghost_zones) = false;
  ADD_ARG(bool, cartesian_vector) = false;

  ADD_ARG(double, x1_slice) = 0.0;
  ADD_ARG(double, x2_slice) = 0.0;
  ADD_ARG(double, x3_slice) = 0.0;

  ADD_ARG(std::string, file_type);
  ADD_ARG(std::string, data_format);
  ADD_ARG(std::vector<std::string>, variables);

  ADD_ARG(bool, combine) = true;
  ADD_ARG(bool, verbose) = false;
};
using OutputOptions = std::shared_ptr<OutputOptionsImpl>;

//! \brief container for output data and metadata; node in nested doubly linked
//! list
struct OutputData {
  std::string type;  // one of (SCALARS,VECTORS) used for vtk outputs
  std::string name;
  std::string longname;
  std::string units;

  AthenaArray<double> data;  // array containing data

  // ptrs to previous and next nodes in doubly linked list:
  OutputData *pnext, *pprev;

  OutputData() : pnext(nullptr), pprev(nullptr) {}
};

//! OutputType is designed to be a node in a singly linked list created & stored
//! in the Output class
class OutputType {
 public:
  OutputOptions options;

  int file_number = 0;
  double next_time = 0.0;

  // constructors
  OutputType() : options(OutputOptionsImpl::create()) {}

  // mark single parameter constructors as "explicit" to prevent them from
  // acting as implicit conversion functions: for f(OutputType arg), prevent
  // f(anOutputParameters)
  explicit OutputType(OutputOptions const &options_);

  // rule of five:
  virtual ~OutputType() = default;
  // copy constructor and assignment operator (pnext_type, pfirst_data, etc. are
  // shallow copied)
  OutputType(const OutputType &copy_other) = default;
  OutputType &operator=(const OutputType &copy_other) = default;
  // move constructor and assignment operator
  OutputType(OutputType &&) = default;
  OutputType &operator=(OutputType &&) = default;

  // data
  // OutputData array start/end index
  int out_is, out_ie, out_js, out_je, out_ks, out_ke;

  // ptr to next node in singly linked list of OutputTypes
  OutputType *pnext_type;

  // functions
  //! \brief Create doubly linked list of OutputData's containing requested
  //! variables
  void LoadOutputData(MeshBlockImpl *pmb, Variables const &vars);

  void AppendOutputDataNode(OutputData *pdata);
  void ReplaceOutputDataNode(OutputData *pold, OutputData *pnew);
  void ClearOutputData();

  bool TransformOutputData(MeshBlockImpl *pmb);

  //! \brief perform data slicing and update the data list
  bool SliceOutputData(MeshBlockImpl *pmb, int dim) { return false; }

  //! \brief perform data summation and update the data list
  void SumOutputData(MeshBlockImpl *pmb, int dim);

  //! \brief Convert vectors in curvilinear coordinates into Cartesian
  void CalculateCartesianVector(torch::Tensor const &src, torch::Tensor dst,
                                Coordinate pco);
  bool ContainVariable(const std::string &var);
  // following pure virtual function must be implemented in all derived classes
  virtual void write_output_file(MeshBlockImpl *pmb, Variables const &vars,
                                 double time, bool flag) {}
  virtual void combine_blocks(MeshBlockImpl *pmb, bool) {}

 protected:
  void loadHydroOutputData(MeshBlockImpl *pmb, Variables const &vars);
  void loadDiagOutputData(MeshBlockImpl *pmb, Variables const &vars);
  void loadScalarOutputData(MeshBlockImpl *pmb, Variables const &vars);
  void loadUserOutputData(MeshBlockImpl *pmb, Variables const &vars);

  int num_vars_;  // number of variables in output
  // nested doubly linked list of OutputData nodes (of the same OutputType):

  // ptr to head OutputData node in doubly linked list
  OutputData *pfirst_data_;

  // ptr to tail OutputData node in doubly linked list
  OutputData *plast_data_;
};
}  // namespace snap

#undef ADD_ARG
