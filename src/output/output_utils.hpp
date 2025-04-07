#pragma once

// C/C++
#include <string>
#include <vector>

// base
#include <configure.h>
#include <fvm/interface/athena_arrays.hpp>

namespace canoe {
int get_num_variables(std::string grid, AthenaArray<Real> const& data);

class MetadataTable {
 protected:
  using StringTable = std::vector<std::vector<std::string>>;

  //! Protected ctor access thru static member function Instance
  MetadataTable();

 public:
  ~MetadataTable();

  static MetadataTable const* GetInstance();

  static void Destroy();

  std::string GetGridType(std::string name) const;

  std::string GetUnits(std::string name) const;

  std::string GetLongName(std::string name) const;

 private:
  StringTable table_;

  //! Pointer to the single MetadataTable instance
  static MetadataTable* myptr_;
};
}  // namespace canoe
