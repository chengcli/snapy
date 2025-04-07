// base
#include <input/parameter_input.hpp>

// fvm
#include "output.hpp"

namespace snap {
void OutputType::loadUserOutputData(MeshBlock pmb) {
  OutputData* pod;

  bool output_all_uov = ContainVariable(options.variable(), "uov") ||
                        ContainVariable(options.variable(), "user_out_var");

  if (!output_all_uov) return;

  for (const auto& pair : pmb->user_out_var) {
    if (pair.key().length() != 0) {
      pod = new OutputData;
      pod->type = "SCALARS";
      pod->name = pair.key();
      pod->data.InitFromTensor(pair.value().unsqueeze(0), 4, 0, 1);
      AppendOutputDataNode(pod);
      num_vars_++;
    }
  }
}
}  // namespace snap
