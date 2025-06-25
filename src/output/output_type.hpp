#pragma once

// C/C++
#include <memory>

// base
#include <configure.h>

// snap
#include <snap/coord/coordinate.hpp>
#include <snap/interface/athena_arrays.hpp>
#include <snap/mesh/meshblock.hpp>

// arg
#include <snap/add_arg.h>

namespace snap {

struct OctTreeOptions {
  ADD_ARG(int, nb1) = 1;  // number of blocks in x1 direction
  ADD_ARG(int, nb2) = 1;  // number of blocks in x2 direction
  ADD_ARG(int, nb3) = 1;  // number of blocks in x3 direction
};

//! \brief  container for parameters read from `<output>` block in the input
struct OutputOptions {
  ADD_ARG(int, fid) = 0;
  ADD_ARG(float, dt) = 0.;
  ADD_ARG(int, dcycle) = 1;

  ADD_ARG(bool, output_slicex1) = false;
  ADD_ARG(bool, output_slicex2) = false;
  ADD_ARG(bool, output_slicex3) = false;

  ADD_ARG(bool, output_sumx1) = false;
  ADD_ARG(bool, output_sumx2) = false;
  ADD_ARG(bool, output_sumx3) = false;

  ADD_ARG(bool, include_ghost_zones) = false;
  ADD_ARG(bool, cartesian_vector) = false;

  ADD_ARG(float, x1_slice) = 0.0;
  ADD_ARG(float, x2_slice) = 0.0;
  ADD_ARG(float, x3_slice) = 0.0;

  ADD_ARG(std::string, block_name);
  ADD_ARG(std::string, file_basename);
  ADD_ARG(std::string, variable);
  ADD_ARG(std::string, file_type);
  ADD_ARG(std::string, data_format);

 public:
  std::string file_id() const { return "out" + std::to_string(fid()); }
};

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
  // control data read from <output> block
  OutputOptions options;

  int file_number = 0;
  double next_time = 0.0;

  // constructors
  OutputType() = default;

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
  void LoadOutputData(MeshBlock pmb);

  void AppendOutputDataNode(OutputData *pdata);
  void ReplaceOutputDataNode(OutputData *pold, OutputData *pnew);
  void ClearOutputData();

  bool TransformOutputData(MeshBlock pmb);

  //! \brief perform data slicing and update the data list
  bool SliceOutputData(MeshBlock pmb, int dim) { return false; }

  //! \brief perform data summation and update the data list
  void SumOutputData(MeshBlock pmb, int dim);

  //! \brief Convert vectors in curvilinear coordinates into Cartesian
  void CalculateCartesianVector(torch::Tensor const &src, torch::Tensor dst,
                                Coordinate pco);
  bool ContainVariable(const std::string &haystack, const std::string &needle);
  // following pure virtual function must be implemented in all derived classes
  virtual void write_output_file(MeshBlock pmb, float time,
                                 OctTreeOptions const &tree, bool flag) {}
  virtual void combine_blocks() {}

 protected:
  void loadHydroOutputData(MeshBlock pmb);
  void loadScalarOutputData(MeshBlock pmb);

  int num_vars_;  // number of variables in output
  // nested doubly linked list of OutputData nodes (of the same OutputType):

  // ptr to head OutputData node in doubly linked list
  OutputData *pfirst_data_;

  // ptr to tail OutputData node in doubly linked list
  OutputData *plast_data_;
};
}  // namespace snap

#undef ADD_ARG
