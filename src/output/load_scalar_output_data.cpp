// snap
#include <snap/snap.h>

#include <snap/scalar/scalar.hpp>

#include "output_type.hpp"

namespace snap {
void OutputType::loadScalarOutputData(MeshBlockImpl* pmb,
                                      Variables const& vars) {
  OutputData* pod;

  if (!vars.count("scalar_r") || !vars.count("scalar_s")) {
    return;
  }

  auto const& r = vars.at("scalar_r");
  auto const& s = vars.at("scalar_s");

  std::string root_name_cons = "s";
  std::string root_name_prim = "r";

  for (int n = 0; n < pmb->pscalar->nvar(); n++) {
    std::string scalar_name_cons, scalar_name_prim;
    scalar_name_cons = root_name_cons + std::to_string(n);
    scalar_name_prim = root_name_prim + std::to_string(n);

    if (ContainVariable(scalar_name_cons) || ContainVariable("cons")) {
      pod = new OutputData;
      pod->type = "SCALARS";
      pod->name = scalar_name_cons;
      pod->data.InitFromTensor(s, 4, n, 1);
      AppendOutputDataNode(pod);
      num_vars_++;
    }

    if (ContainVariable(scalar_name_prim) || ContainVariable("prim")) {
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
