// snap
#include "output_type.hpp"

namespace snap {
void OutputType::loadUserOutputData(MeshBlockImpl* pmb, Variables const& vars) {
  OutputData* pod;

  bool output_all_uov =
      ContainVariable("uov") || ContainVariable("user_out_var");

  if (!output_all_uov) return;

  auto user_out_var = pmb->user_output_callback(vars);

  for (const auto& pair : user_out_var) {
    if (pair.first.length() != 0) {
      pod = new OutputData;
      pod->type = "SCALARS";
      pod->name = pair.first;
      pod->data.CopyFromTensor(pair.second);
      AppendOutputDataNode(pod);
      num_vars_++;
    }
  }
}
}  // namespace snap
