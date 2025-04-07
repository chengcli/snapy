// C/C++ headers
#include <sstream>
#include <stdexcept>

// base
#include <configure.h>

// snap
#include "output.hpp"
#include "output_formats.hpp"

namespace snap {
OutputType::OutputType(OutputOptions const &options_)
    : options(options_),
      pnext_type(),    // Terminate this node in singly linked list with nullptr
      num_vars_(),     // nested doubly linked list of OutputData:
      pfirst_data_(),  // Initialize head node to nullptr
      plast_data_() {  // Initialize tail node to nullptr
}

void OutputType::LoadOutputData(MeshBlock pmb) {
  num_vars_ = 0;
  OutputData *pod;

  loadHydroOutputData(pmb);
  loadScalarOutputData(pmb);
  loadUserOutputData(pmb);

  // throw an error if output variable name not recognized
  if (num_vars_ == 0) {
    std::stringstream msg;
    msg << "### FATAL ERROR in function [OutputType::LoadOutputData]"
        << std::endl
        << "Output variable '" << options.variable() << "' not implemented"
        << std::endl;
    throw std::runtime_error(msg.str());
  }

  return;
}

void OutputType::AppendOutputDataNode(OutputData *pnew_data) {
  if (pfirst_data_ == nullptr) {
    pfirst_data_ = pnew_data;
  } else {
    pnew_data->pprev = plast_data_;
    plast_data_->pnext = pnew_data;
  }
  // make the input node the new tail node of the doubly linked list
  plast_data_ = pnew_data;
}

void OutputType::ReplaceOutputDataNode(OutputData *pold, OutputData *pnew) {
  if (pold == pfirst_data_) {
    pfirst_data_ = pnew;
    if (pold->pnext != nullptr) {  // there is another node in the list
      pnew->pnext = pold->pnext;
      pnew->pnext->pprev = pnew;
    } else {  // there is only one node in the list
      plast_data_ = pnew;
    }
  } else if (pold == plast_data_) {
    plast_data_ = pnew;
    pnew->pprev = pold->pprev;
    pnew->pprev->pnext = pnew;
  } else {
    pnew->pnext = pold->pnext;
    pnew->pprev = pold->pprev;
    pnew->pprev->pnext = pnew;
    pnew->pnext->pprev = pnew;
  }
  delete pold;
}

void OutputType::ClearOutputData() {
  OutputData *pdata = pfirst_data_;
  while (pdata != nullptr) {
    OutputData *pdata_old = pdata;
    pdata = pdata->pnext;
    delete pdata_old;
  }
  // reset pointers to head and tail nodes of doubly linked list:
  pfirst_data_ = nullptr;
  plast_data_ = nullptr;
}

bool OutputType::TransformOutputData(MeshBlock pmb) {
  bool flag = true;
  if (options.output_slicex3()) {
    bool ret = SliceOutputData(pmb, 3);
    if (!ret) flag = false;
  }
  if (options.output_slicex2()) {
    bool ret = SliceOutputData(pmb, 2);
    if (!ret) flag = false;
  }
  if (options.output_slicex1()) {
    bool ret = SliceOutputData(pmb, 1);
    if (!ret) flag = false;
  }
  if (options.output_sumx3()) {
    SumOutputData(pmb, 3);
  }
  if (options.output_sumx2()) {
    SumOutputData(pmb, 2);
  }
  if (options.output_sumx1()) {
    SumOutputData(pmb, 1);
  }
  return flag;
}

bool OutputType::ContainVariable(const std::string &haystack,
                                 const std::string &needle) {
  if (haystack.compare(needle) == 0) return true;
  if (haystack.find(',' + needle + ',') != std::string::npos) return true;
  if (haystack.find(needle + ',') == 0) return true;
  if (haystack.find(',' + needle) != std::string::npos &&
      haystack.find(',' + needle) == haystack.length() - needle.length() - 1)
    return true;
  return false;
}

}  // namespace snap
