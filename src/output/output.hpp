#pragma once

// C/C++ headers
#include <cstdio>
#include <string>

// torch
#include <torch/nn/module.h>

// base
#include <configure.h>
#include <input/parameter_input.hpp>

// fvm
#include <fvm/mesh/mesh.hpp>
#include "output_type.hpp"

namespace canoe {
//! \brief root class for all Athena++ outputs. Provides a singly linked list of
//! OutputTypes, with each node representing one mode/format of output to be
//! made.
class OutputImpl : public torch::nn::Module {
 public:
  OutputImpl(Mesh pm, ParameterInput pin);
  ~OutputImpl();

  void MakeOutput(Mesh pm, ParameterInput pin, bool wtflag = false);

 private:
  // ptr to head OutputType node in singly linked list
  // (not storing a reference to the tail node)
  OutputType *pfirst_type_;
};
TORCH_MODULE(Output);

}  // namespace canoe
