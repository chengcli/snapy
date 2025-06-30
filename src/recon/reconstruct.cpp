// yaml
#include <yaml-cpp/yaml.h>

// snap
#include <snap/snap.h>

#include <snap/registry.hpp>

#include "reconstruct.hpp"

namespace snap {
ReconstructOptions ReconstructOptions::from_yaml(const YAML::Node &node,
                                                 std::string section) {
  ReconstructOptions op;

  if (!node[section]) {
    TORCH_WARN("no section '", section,
               "' specified, using default reconstruction model");
    return op;
  }

  op.shock() = node[section]["shock"].as<bool>(false);
  op.interp().type() = node[section]["type"].as<std::string>("dc");
  op.interp().scale() = node[section]["scale"].as<bool>(false);

  return op;
}

// TODO(cli) remove copy
void _apply_inplace(int dim, int il, int iu, const torch::Tensor &w,
                    Interp &pinterp, torch::Tensor wlr) {
  if (il > iu) return;

  auto result = pinterp->forward(w, dim);

  wlr[Index::ILT].slice(dim, il, iu + 1) =
      result[Index::IRT].narrow(dim, 0, iu - il + 1);
  wlr[Index::IRT].slice(dim, il, iu + 1) =
      result[Index::ILT].narrow(dim, 1, iu - il + 1);
}

ReconstructImpl::ReconstructImpl(const ReconstructOptions &options_)
    : options(options_) {
  reset();
}

void ReconstructImpl::reset() {
  pinterp1 = register_module_op(this, "interp1", options.interp());
  pinterp2 = register_module_op(this, "interp2", options.interp());
}

torch::Tensor ReconstructImpl::forward(torch::Tensor w, int dim) {
  torch::NoGradGuard no_grad;

  auto vec = w.sizes().vec();
  vec.insert(vec.begin(), 2);

  auto result = torch::zeros(vec, w.options());

  auto dim_size = w.size(dim);
  int nghost = pinterp1->stencils() / 2 + 1;
  int il = nghost;
  int iu = dim_size - nghost;

  TORCH_CHECK(il <= iu, "il > iu");

  if (options.shock()) {
    _apply_inplace(dim, il, iu, w, pinterp1, result);
    return result;
  }

  /* modify velocity/pressure variables
  if (dim_size > 2 * nghost) {
    if (options.is_boundary_lower()) {
      il += nghost;
    } else if (options.is_boundary_upper()) {
      iu -= nghost;
    }
  } else {
    if (options.is_boundary_lower() && !options.is_boundary_upper()) {
      il += nghost;
    } else if (!options.is_boundary_lower() && options.is_boundary_upper()) {
      iu -= nghost;
    } else if (options.is_boundary_lower() && options.is_boundary_upper()) {
      int len1 = dim_size / 2;
      int len2 = dim_size - len1;
      il += len1;
      iu -= len2;
    }
  }

  // interior
  auto w_ = w.narrow(0, index::IVX, 4);
  auto wlr_ = result.narrow(1, index::IVX, 4);
  _apply_inplace(dim, il, iu, w_, pinterp2, wlr_);*/

  // density
  _apply_inplace(dim, il, iu, w.narrow(0, Index::IDN, 1), pinterp1,
                 result.narrow(1, Index::IDN, 1));

  // velocity/pressure
  _apply_inplace(dim, il, iu, w.narrow(0, Index::IVX, 4), pinterp2,
                 result.narrow(1, Index::IVX, 4));

  // others
  int ny = w.size(0) - 5;
  _apply_inplace(dim, il, iu, w.narrow(0, Index::ICY, ny), pinterp1,
                 result.narrow(1, Index::ICY, ny));

  return result;
}
}  // namespace snap
