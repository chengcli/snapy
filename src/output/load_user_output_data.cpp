// snap
#include <snap/mesh/meshblock.hpp>

#include "output_type.hpp"

namespace snap {
void OutputType::loadUserOutputData(MeshBlockImpl* pmb, Variables const& vars) {
  OutputData* pod;

  bool output_all_uov =
      ContainVariable("uov") || ContainVariable("user_out_var");

  if (!output_all_uov) return;

  TORCH_CHECK(
      static_cast<bool>(pmb->user_output_callback),
      "Output requested 'uov' or 'user_out_var', but no user output callback "
      "was registered. Call set_user_output_func(...) before make_outputs(), "
      "or remove 'uov' from the output variable list.");

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
