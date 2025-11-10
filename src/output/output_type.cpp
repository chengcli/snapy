// C/C++ headers
#include <sstream>
#include <stdexcept>

// snap
#include "output_formats.hpp"
#include "output_type.hpp"

namespace snap {
OutputOptions OutputOptions::from_yaml(YAML::Node const &node, int fid) {
  OutputOptions options;

  options.fid() = fid;
  options.dt() = node["dt"].as<double>(0.);

  options.output_slicex1() = node["output_slicex1"].as<bool>(false);
  options.output_slicex2() = node["output_slicex2"].as<bool>(false);
  options.output_slicex3() = node["output_slicex3"].as<bool>(false);

  options.output_sumx1() = node["output_sumx1"].as<bool>(false);
  options.output_sumx2() = node["output_sumx2"].as<bool>(false);
  options.output_sumx3() = node["output_sumx3"].as<bool>(false);

  options.include_ghost_zones() = node["include_ghost_zones"].as<bool>(false);
  options.cartesian_vector() = node["cartesian_vector"].as<bool>(false);

  options.x1_slice() = node["x1_slice"].as<double>(0.);
  options.x2_slice() = node["x2_slice"].as<double>(0.);
  options.x3_slice() = node["x3_slice"].as<double>(0.);

  if (node["type"]) {
    options.file_type() = node["type"].as<std::string>();
  } else {
    throw std::invalid_argument(
        "OutputOptions::from_yaml: output file type "
        "must be specified");
  }

  if (node["data_format"]) {
    options.data_format() = node["data_format"].as<std::string>();
  }

  if (node["variables"]) {
    options.variables() = node["variables"].as<std::vector<std::string>>();
  }

  return options;
}

OutputType::OutputType(OutputOptions const &options_)
    : options(options_),
      pnext_type(),    // Terminate this node in singly linked list with nullptr
      num_vars_(),     // nested doubly linked list of OutputData:
      pfirst_data_(),  // Initialize head node to nullptr
      plast_data_() {  // Initialize tail node to nullptr
}

void OutputType::LoadOutputData(MeshBlockImpl *pmb, Variables const &vars) {
  num_vars_ = 0;
  OutputData *pod;

  loadHydroOutputData(pmb, vars);
  loadDiagOutputData(pmb, vars);
  loadScalarOutputData(pmb, vars);
  loadUserOutputData(pmb, vars);

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

bool OutputType::ContainVariable(const std::string &var) {
  return std::find(options.variables().begin(), options.variables().end(),
                   var) != options.variables().end();
}

}  // namespace snap
