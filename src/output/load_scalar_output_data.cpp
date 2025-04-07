// fmt
#include <spdlog/spdlog.h>

// base
#include <globals.h>

// fvm
#include <fvm/scalar/scalar.hpp>
#include "output.hpp"

namespace canoe {
void OutputType::loadScalarOutputData(MeshBlock pmb) {
  OutputData *pod;

  std::string root_name_cons = "s";
  std::string root_name_prim = "r";

  for (int n = 0; n < pmb->pscalar->nvar(); n++) {
    std::string scalar_name_cons, scalar_name_prim;
    scalar_name_cons = root_name_cons + std::to_string(n);
    scalar_name_prim = root_name_prim + std::to_string(n);

    if (ContainVariable(options.variable(), scalar_name_cons) ||
        ContainVariable(options.variable(), "cons")) {
      pod = new OutputData;
      pod->type = "SCALARS";
      pod->name = scalar_name_cons;
      pod->data.InitFromTensor(pmb->scalar_u, 4, n, 1);
      AppendOutputDataNode(pod);
      num_vars_++;
    }

    if (ContainVariable(options.variable(), scalar_name_prim) ||
        ContainVariable(options.variable(), "prim")) {
      pod = new OutputData;
      pod->type = "SCALARS";
      pod->name = scalar_name_prim;
      pod->data.InitFromTensor(GET_SHARED("scalar/w"), 4, n, 1);
      AppendOutputDataNode(pod);
      num_vars_++;
    }
  }
}
}  // namespace canoe
