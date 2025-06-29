// yaml
#include <yaml-cpp/yaml.h>

// snap
#include <snap/bc/bc_func.hpp>

#include "meshblock.hpp"

namespace snap {

MeshBlockOptions MeshBlockOptions::from_yaml(std::string input_file) {
  MeshBlockOptions op;

  op.hydro() = HydroOptions::from_yaml(input_file);
  op.intg() = IntegratorOptions::from_yaml(input_file);

  auto config = YAML::LoadFile(input_file);

  if (!config["boundary-condition"]) return op;
  if (!config["boundary-condition"]["external"]) return op;

  auto external_bc = config["boundary-condition"]["external"];

  if (op.hydro().coord().nc1() > 1) {
    // x1-inner
    auto ix1 = external_bc["x1-inner"].as<std::string>("reflecting");
    ix1 += "_inner";
    TORCH_CHECK(get_bc_func().find(ix1) != get_bc_func().end(),
                "Boundary function '", ix1, "' is not defined.");
    op.bfuncs().push_back(get_bc_func()[ix1]);

    // x1-outer
    auto ox1 = external_bc["x1-outer"].as<std::string>("reflecting");
    ox1 += "_outer";
    TORCH_CHECK(get_bc_func().find(ox1) != get_bc_func().end(),
                "Boundary function '", ox1, "' is not defined.");
    op.bfuncs().push_back(get_bc_func()[ox1]);
  } else if (op.hydro().coord().nc2() > 1 || op.hydro().coord().nc3() > 1) {
    op.bfuncs().push_back(get_bc_func()["exchange_inner"]);  // null-op
    op.bfuncs().push_back(get_bc_func()["exchange_outer"]);  // null-op
  }

  if (op.hydro().coord().nc2() > 1) {
    // x2-inner
    auto ix2 = external_bc["x2-inner"].as<std::string>("reflecting");
    ix2 += "_inner";
    TORCH_CHECK(get_bc_func().find(ix2) != get_bc_func().end(),
                "Boundary function '", ix2, "' is not defined.");
    op.bfuncs().push_back(get_bc_func()[ix2]);

    // x2-outer
    auto ox2 = external_bc["x2-outer"].as<std::string>("reflecting");
    ox2 += "_outer";
    TORCH_CHECK(get_bc_func().find(ox2) != get_bc_func().end(),
                "Boundary function '", ox2, "' is not defined.");
    op.bfuncs().push_back(get_bc_func()[ox2]);
  } else if (op.hydro().coord().nc3() > 1) {
    op.bfuncs().push_back(get_bc_func()["exchange_inner"]);  // null-op
    op.bfuncs().push_back(get_bc_func()["exchange_outer"]);  // null-op
  }

  if (op.hydro().coord().nc3() > 1) {
    // x3-inner
    auto ix3 = external_bc["x3-inner"].as<std::string>("reflecting");
    ix3 += "_inner";
    TORCH_CHECK(get_bc_func().find(ix3) != get_bc_func().end(),
                "Boundary function '", ix3, "' is not defined.");
    op.bfuncs().push_back(get_bc_func()[ix3]);

    // x3-outer
    auto ox3 = external_bc["x3-outer"].as<std::string>("reflecting");
    ox3 += "_outer";
    TORCH_CHECK(get_bc_func().find(ox3) != get_bc_func().end(),
                "Boundary function '", ox3, "' is not defined.");
    op.bfuncs().push_back(get_bc_func()[ox3]);
  }

  return op;
}

}  // namespace snap
