// snap
#include <snap/snap.h>

#include <snap/mesh/meshblock.hpp>
#include <snap/scalar/scalar.hpp>

#include "output_type.hpp"

namespace snap {
void OutputType::loadScalarOutputData(MeshBlockImpl* pmb,
                                      Variables const& vars) {
  OutputData* pod;

  if (!vars.count("scalar_r") && !vars.count("scalar_s")) {
    return;
  }

  auto const& r =
      vars.count("scalar_r") ? vars.at("scalar_r") : torch::Tensor();
  auto const& s =
      vars.count("scalar_s") ? vars.at("scalar_s") : torch::Tensor();
  bool output_all_scalar = ContainVariable("scalar");
  bool output_all_scalar_cons =
      output_all_scalar || ContainVariable("scalar_cons");
  bool output_all_scalar_prim =
      output_all_scalar || ContainVariable("scalar_prim");

  std::string root_name_cons = "s";
  std::string root_name_prim = "r";

  for (int n = 0; n < pmb->pscalar->nvar(); n++) {
    std::string scalar_name_cons, scalar_name_prim;
    auto const& names = pmb->pscalar->options->names();
    if (n < names.size() && !names[n].empty()) {
      scalar_name_cons = root_name_cons + "_" + names[n];
      scalar_name_prim = root_name_prim + "_" + names[n];
    } else {
      scalar_name_cons = root_name_cons + std::to_string(n);
      scalar_name_prim = root_name_prim + std::to_string(n);
    }

    if (s.defined() && (ContainVariable(scalar_name_cons) ||
                        ContainVariable("cons") || output_all_scalar_cons)) {
      pod = new OutputData;
      pod->type = "SCALARS";
      pod->name = scalar_name_cons;
      pod->data.InitFromTensor(s, 4, n, 1);
      AppendOutputDataNode(pod);
      num_vars_++;
    }

    if (r.defined() && (ContainVariable(scalar_name_prim) ||
                        ContainVariable("prim") || output_all_scalar_prim)) {
      pod = new OutputData;
      pod->type = "SCALARS";
      pod->name = scalar_name_prim;
      pod->data.InitFromTensor(r, 4, n, 1);
      AppendOutputDataNode(pod);
      num_vars_++;
    }
  }
}
}  // namespace snap
