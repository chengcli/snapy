#pragma once

// C/C++
#include <string>
#include <vector>

// snap
#include "output_type.hpp"

namespace snap {
/*! \brief derived OutputType class for history dumps
class HistoryOutput : public OutputType {
 public:
  explicit HistoryOutput(OutputParameters oparams) : OutputType(oparams) {}
  void write_output_file(Mesh pm, ParameterInput pin, bool flag) override;
};

//! \brief derived OutputType class for formatted table (tabular) data
class FormattedTableOutput : public OutputType {
 public:
  explicit FormattedTableOutput(OutputParameters oparams)
      : OutputType(oparams) {}
  void write_output_file(Mesh pm, ParameterInput pin, bool flag) override;
};

//! \brief derived OutputType class for vtk dumps
class VTKOutput : public OutputType {
 public:
  explicit VTKOutput(OutputParameters oparams) : OutputType(oparams) {}
  void write_output_file(Mesh pm, ParameterInput pin, bool flag) override;
};*/

//! \brief derived OutputType class for restart dumps
class RestartOutput : public OutputType {
 public:
  explicit RestartOutput(OutputOptions const& options_);
  void write_output_file(MeshBlockImpl* pmb, Variables const& vars, double time,
                         bool final_write) override;
};

// \brief derived OutputType class for Athena HDF5 files
class HDF5Output : public OutputType {
 public:
  // Function declarations
  explicit HDF5Output(OutputOptions const& options_);
  void write_output_file(MeshBlockImpl* pmb, Variables const& var, double time,
                         bool final_write) override;
  void MakeXDMF();

 private:
  // Parameters
  // maximum length of names excluding \0
  static const int max_name_length = 20;

  // Metadata
  std::string filename;   // name of athdf file
  double code_time;       // time in code unit for XDMF
  int num_blocks_global;  // number of MeshBlocks in simulation
  int nx1, nx2, nx3;      // sizes of MeshBlocks
  int num_datasets;       // count of datasets to output
  int* num_variables;     // list of counts of variables per dataset

  // array of C-string names of datasets
  char (*dataset_names)[max_name_length + 1];

  // array of C-string names of variables
  char (*variable_names)[max_name_length + 1];
};

/*class DebugOutput : public OutputType {
 public:
  explicit DebugOutput(OutputParameters oparams) : OutputType(oparams) {}
  void write_output_file(Mesh pm, ParameterInput pin, bool flag) override;
};*/

class NetcdfOutput : public OutputType {
 public:
  explicit NetcdfOutput(OutputOptions const& options_);
  ~NetcdfOutput() {}

  //!  \brief Cycles over all MeshBlocks and writes OutputData in NETCDF format,
  ///         one MeshBlock per file
  void write_output_file(MeshBlockImpl* pmb, Variables const& vars, double time,
                         bool final_write) override;
};

class PnetcdfOutput : public OutputType {
 public:
  explicit PnetcdfOutput(OutputOptions const& options_);
  ~PnetcdfOutput() {}
  void write_output_file(MeshBlockImpl* pmb, Variables const& vars, double time,
                         bool final_write) override;
};

class FITSOutput : public OutputType {
 public:
  explicit FITSOutput(OutputOptions const& options_);
  ~FITSOutput() {}
  void write_output_file(MeshBlockImpl* pmb, Variables const& vars, double time,
                         bool final_write) override;
};
}  // namespace snap
