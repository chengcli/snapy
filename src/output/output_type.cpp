// C/C++ headers
#include <sstream>
#include <stdexcept>

// snap
#include <snap/mesh/meshblock.hpp>
#include <snap/utils/refine.hpp>

#include "output_formats.hpp"
#include "output_type.hpp"

namespace snap {
OutputOptions OutputOptionsImpl::from_yaml(YAML::Node const& node, int fid) {
  auto options = OutputOptionsImpl::create();

  options->fid() = fid;
  options->dt() = node["dt"].as<double>(0.);

  options->output_slicex1() = node["output_slicex1"].as<bool>(false);
  options->output_slicex2() = node["output_slicex2"].as<bool>(false);
  options->output_slicex3() = node["output_slicex3"].as<bool>(false);

  options->output_sumx1() = node["output_sumx1"].as<bool>(false);
  options->output_sumx2() = node["output_sumx2"].as<bool>(false);
  options->output_sumx3() = node["output_sumx3"].as<bool>(false);

  options->include_ghost_zones() = node["include_ghost_zones"].as<bool>(false);
  options->cartesian_vector() = node["cartesian_vector"].as<bool>(false);

  options->x1_slice() = node["x1_slice"].as<double>(0.);
  options->x2_slice() = node["x2_slice"].as<double>(0.);
  options->x3_slice() = node["x3_slice"].as<double>(0.);

  if (node["type"]) {
    options->file_type() = node["type"].as<std::string>();
  } else {
    throw std::invalid_argument(
        "OutputOptions::from_yaml: output file type "
        "must be specified");
  }

  if (node["data_format"]) {
    options->data_format() = node["data_format"].as<std::string>();
  }

  if (node["variables"]) {
    options->variables() = node["variables"].as<std::vector<std::string>>();
  }

  if (node["combine"]) {
    options->combine() = node["combine"].as<bool>(true);
  }

  options->verbose() = node["verbose"].as<bool>(false);
  options->super_resolution() = node["super-resolution"].as<bool>(false);

  return options;
}

OutputType::OutputType(OutputOptions const& options_)
    : options(options_),
      pnext_type(),    // Terminate this node in singly linked list with nullptr
      num_vars_(),     // nested doubly linked list of OutputData:
      pfirst_data_(),  // Initialize head node to nullptr
      plast_data_() {  // Initialize tail node to nullptr
}

MeshBlockImpl* OutputType::LoadOutputData(MeshBlockImpl* pmb_in,
                                          Variables const& vars_in) {
  num_vars_ = 0;
  OutputData* pod;
  MeshBlockImpl* pmb;
  Variables vars;
  // set ptrs to data in OutputData linked list, then slice/sum as needed

  // create a refined output meshblock if super resolution is requested
  if (options->super_resolution()) {
    auto op = std::make_shared<MeshBlockOptionsImpl>(*(pmb_in->options));
    op->coord() = pmb_in->options->coord()->clone();
    if (op->coord()->nx2() > 1) op->coord()->nx2() *= 2;
    if (op->coord()->nx3() > 1) op->coord()->nx3() *= 2;

    pmb = new MeshBlockImpl(op);
    // shall be deleted by caller of LoadOutputData

    auto peos = pmb->phydro->peos;
    auto pscalar = pmb->pscalar;

    int nghost = pmb->options->coord()->nghost();

    for (auto& [name, var] : vars_in) {
      auto interior_in = pmb_in->part(
          {0, 0, 0}, PartOptions().exterior(false).ndim(var.dim()));
      auto interior_out =
          pmb->part({0, 0, 0}, PartOptions().exterior(false).ndim(var.dim()));
      auto vec = var.sizes().vec();

      // dim 2
      if (vec.size() > 1 && vec[vec.size() - 2] > 1) {
        vec[vec.size() - 2] = 2 * (vec[vec.size() - 2] - nghost);
      }

      // dim 3
      if (vec.size() > 2 && vec[vec.size() - 3] > 1) {
        vec[vec.size() - 3] = 2 * (vec[vec.size() - 3] - nghost);
      }

      vars[name] = torch::zeros(vec, var.options());
      vars[name]
          .index(interior_out)
          .copy_(conservative_refine(var.index(interior_in)));
    }
    // vars["hydro_w"] = peos->compute("U->W", {vars["hydro_u"]});
    // scalar eos?
  } else {
    pmb = pmb_in;
    vars = vars_in;
  }

  loadHydroOutputData(pmb, vars);
  loadDiagOutputData(pmb, vars);
  loadScalarOutputData(pmb, vars);
  loadUserOutputData(pmb_in, vars);

  return pmb;
}

void OutputType::AppendOutputDataNode(OutputData* pnew_data) {
  if (pfirst_data_ == nullptr) {
    pfirst_data_ = pnew_data;
  } else {
    pnew_data->pprev = plast_data_;
    plast_data_->pnext = pnew_data;
  }
  // make the input node the new tail node of the doubly linked list
  plast_data_ = pnew_data;
}

void OutputType::ReplaceOutputDataNode(OutputData* pold, OutputData* pnew) {
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
  OutputData* pdata = pfirst_data_;
  while (pdata != nullptr) {
    OutputData* pdata_old = pdata;
    pdata = pdata->pnext;
    delete pdata_old;
  }
  // reset pointers to head and tail nodes of doubly linked list:
  pfirst_data_ = nullptr;
  plast_data_ = nullptr;
}

bool OutputType::ContainAnyVariable(
    std::initializer_list<std::string> vars) const {
  for (auto const& var : vars) {
    if (std::find(options->variables().begin(), options->variables().end(),
                  var) != options->variables().end()) {
      return true;
    }
  }
  return false;
}

void OutputType::AccumulatePrimStat(Variables const& vars,
                                    double current_time) {
  if (!OutputsAnyStat()) return;
  if (!prim_stat_initialized_) {
    prim_stat_last_time_ = current_time;
    prim_stat_initialized_ = true;
    return;
  }

  double dt = current_time - prim_stat_last_time_;
  if (dt > 0.0) {
    if (OutputsPrimStat()) {
      auto it = vars.find("hydro_w");
      if (it != vars.end() && it->second.defined()) {
        auto const& w = it->second;
        if (!prim_stat_sum_.defined()) {
          prim_stat_sum_ = torch::zeros_like(w);
          prim_stat_sum_sq_ = torch::zeros_like(w);
        }
        prim_stat_sum_.add_(w * dt);
        prim_stat_sum_sq_.add_(w * w * dt);
      }
    }

    if (OutputsScalarStat()) {
      auto it = vars.find("scalar_r");
      if (it != vars.end() && it->second.defined()) {
        auto const& r = it->second;
        if (!scalar_stat_sum_.defined()) {
          scalar_stat_sum_ = torch::zeros_like(r);
          scalar_stat_sum_sq_ = torch::zeros_like(r);
        }
        scalar_stat_sum_.add_(r * dt);
        scalar_stat_sum_sq_.add_(r * r * dt);
      }
    }

    prim_stat_elapsed_ += dt;
  }
  prim_stat_last_time_ = current_time;
}

void OutputType::ResetPrimStat(double current_time) {
  if (!OutputsAnyStat()) return;
  if (prim_stat_sum_.defined()) {
    prim_stat_sum_.zero_();
  }
  if (prim_stat_sum_sq_.defined()) {
    prim_stat_sum_sq_.zero_();
  }
  if (scalar_stat_sum_.defined()) {
    scalar_stat_sum_.zero_();
  }
  if (scalar_stat_sum_sq_.defined()) {
    scalar_stat_sum_sq_.zero_();
  }
  prim_stat_elapsed_ = 0.0;
  prim_stat_last_time_ = current_time;
  prim_stat_initialized_ = true;
}

torch::Tensor OutputType::PrimStatMean(torch::Tensor const& current) const {
  if (prim_stat_elapsed_ <= 0.0 || !prim_stat_sum_.defined()) {
    return current;
  }
  return prim_stat_sum_ / prim_stat_elapsed_;
}

torch::Tensor OutputType::PrimStatStd(torch::Tensor const& current) const {
  if (prim_stat_elapsed_ <= 0.0 || !prim_stat_sum_sq_.defined()) {
    return torch::zeros_like(current);
  }
  auto mean = prim_stat_sum_ / prim_stat_elapsed_;
  auto variance = prim_stat_sum_sq_ / prim_stat_elapsed_ - mean * mean;
  return torch::sqrt(torch::clamp_min(variance, 0.0));
}

torch::Tensor OutputType::ScalarStatMean(torch::Tensor const& current) const {
  if (prim_stat_elapsed_ <= 0.0 || !scalar_stat_sum_.defined()) {
    return current;
  }
  return scalar_stat_sum_ / prim_stat_elapsed_;
}

torch::Tensor OutputType::ScalarStatStd(torch::Tensor const& current) const {
  if (prim_stat_elapsed_ <= 0.0 || !scalar_stat_sum_sq_.defined()) {
    return torch::zeros_like(current);
  }
  auto mean = scalar_stat_sum_ / prim_stat_elapsed_;
  auto variance = scalar_stat_sum_sq_ / prim_stat_elapsed_ - mean * mean;
  return torch::sqrt(torch::clamp_min(variance, 0.0));
}

void OutputType::appendTensorOutput(std::string type, std::string name,
                                    torch::Tensor const& tensor) {
  auto* pod = new OutputData;
  pod->type = std::move(type);
  pod->name = std::move(name);
  pod->data.CopyFromTensor(tensor);
  AppendOutputDataNode(pod);
  num_vars_++;
}

void OutputType::appendTensorSliceOutput(std::string type, std::string name,
                                         torch::Tensor const& tensor, int dim,
                                         int start, int count) {
  auto* pod = new OutputData;
  pod->type = std::move(type);
  pod->name = std::move(name);
  pod->data.InitFromTensor(tensor, dim, start, count);
  AppendOutputDataNode(pod);
  num_vars_ += count;
}

}  // namespace snap
