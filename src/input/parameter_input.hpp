#pragma once

// C/C++
#include <cstddef>  // std::size_t
#include <memory>   // shared_ptr
#include <ostream>  // ostream
#include <string>   // string

// snap
#include <configure.h>

#include "io_wrapper.hpp"

// OpenMP
#ifdef OPENMP_PARALLEL
#include <omp.h>
#endif

namespace snap {
//! \brief  node in a singly linked list of parameters contained within 1x input
//! block
struct InputLine {
  std::string param_name;
  std::string param_value;  // value of the parameter is stored as a string!
  std::string param_comment;

  // pointer to the next node in this nested singly linked list
  InputLine* pnext;
};

//! \brief node in a singly linked list of all input blocks contained within
//! input file
class InputBlock {
 public:
  InputBlock() = default;
  ~InputBlock();

  // data
  std::string block_name;

  // length of longest param_name, for nice-looking output
  std::size_t max_len_parname;

  // length of longest param_value, to format outputs
  std::size_t max_len_parvalue;

  // pointer to the next node in InputBlock singly linked list
  InputBlock* pnext;

  // pointer to head node in nested singly linked list (in this block)
  InputLine* pline;

  // (not storing a reference to the tail node)

  // functions
  InputLine* GetPtrToLine(std::string name);
};

//! \brief data and definitions of functions used to store and access input
//! parameters
//!
//! Functions are implemented in parameter_input.cpp
class ParameterInputImpl {
 public:
  using Real = double;

  // constructor/destructor
  ParameterInputImpl();
  ~ParameterInputImpl();

  // data
  // pointer to head node in singly linked list of InputBlock
  // (not storing a reference to the tail node)
  InputBlock* pfirst_block;

  // functions
  void LoadFromStream(std::istream& is);
  void LoadFromFile(IOWrapper& input);
  void ModifyFromCmdline(int argc, char* argv[]);
  void ParameterDump(std::ostream& os);
  int DoesParameterExist(std::string block, std::string name);
  int GetInteger(std::string block, std::string name);
  int GetOrAddInteger(std::string block, std::string name, int value);
  int SetInteger(std::string block, std::string name, int value);
  Real GetReal(std::string block, std::string name);
  Real GetOrAddReal(std::string block, std::string name, Real value);
  Real SetReal(std::string block, std::string name, Real value);
  bool GetBoolean(std::string block, std::string name);
  bool GetOrAddBoolean(std::string block, std::string name, bool value);
  bool SetBoolean(std::string block, std::string name, bool value);
  std::string GetString(std::string block, std::string name);
  std::string GetOrAddString(std::string block, std::string name,
                             std::string value);
  std::string SetString(std::string block, std::string name, std::string value);
  void RollbackNextTime();
  void ForwardNextTime(Real time);

  InputBlock* GetPtrToBlock(std::string name);
  void AddParameter(InputBlock* pib, std::string name, std::string value,
                    std::string comment);

 private:
  // last input file opened, to prevent duplicate reads
  std::string last_filename_;

  InputBlock* FindOrAddBlock(std::string name);
  void ParseLine(InputBlock* pib, std::string line, std::string& name,
                 std::string& value, std::string& comment);

  // thread safety
#ifdef OPENMP_PARALLEL
  omp_lock_t lock_;
#endif

  void Lock();
  void Unlock();
};

using ParameterInput = std::shared_ptr<ParameterInputImpl>;
}  // namespace snap
