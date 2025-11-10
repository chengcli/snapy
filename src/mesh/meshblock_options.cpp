// yaml
#include <yaml-cpp/yaml.h>

// snap
#include <snap/output/output_type.hpp>

#include "meshblock.hpp"

namespace snap {

MeshBlockOptions MeshBlockOptions::from_yaml(std::string input_file,
                                             DistributeInfo _dist) {
  MeshBlockOptions op;

  // use the basename of the input file as the basename of the output files
  op.basename() = input_file.substr(0, input_file.find_last_of('.'));

  op.dist() = _dist;
  op.hydro() = HydroOptions::from_yaml(input_file, op.dist());
  op.intg() = IntegratorOptions::from_yaml(input_file);

  auto config = YAML::LoadFile(input_file);

  // --------------- boundary conditions --------------- //

  if (!config["boundary-condition"]) return op;
  if (!config["boundary-condition"]["external"]) return op;

  auto external_bc = config["boundary-condition"]["external"];

  if (op.hydro().coord().nc1() > 1) {
    // x1-inner
    auto ix1 = external_bc["x1-inner"].as<std::string>("reflecting");
    ix1 += "_inner";
    TORCH_CHECK(get_bc_func().find(ix1) != get_bc_func().end(),
                "Boundary function '", ix1, "' is not defined.");

    if (op.dist().lx1() == 0) {  // physical boundary
      op.bfuncs().push_back(get_bc_func()[ix1]);
    } else {  // block boundary
      op.bfuncs().push_back(nullptr);
    }

    // x1-outer
    auto ox1 = external_bc["x1-outer"].as<std::string>("reflecting");
    ox1 += "_outer";
    TORCH_CHECK(get_bc_func().find(ox1) != get_bc_func().end(),
                "Boundary function '", ox1, "' is not defined.");

    if (op.dist().lx1() == op.dist().nb1() - 1) {  // physical boundary
      op.bfuncs().push_back(get_bc_func()[ox1]);
    } else {  // block boundary
      op.bfuncs().push_back(nullptr);
    }
  } else if (op.hydro().coord().nc2() > 1 || op.hydro().coord().nc3() > 1) {
    op.bfuncs().push_back(nullptr);
    op.bfuncs().push_back(nullptr);
  }

  if (op.hydro().coord().nc2() > 1) {
    // x2-inner
    auto ix2 = external_bc["x2-inner"].as<std::string>("reflecting");
    ix2 += "_inner";
    TORCH_CHECK(get_bc_func().find(ix2) != get_bc_func().end(),
                "Boundary function '", ix2, "' is not defined.");

    if (op.dist().lx2() == 0) {  // physical boundary
      op.bfuncs().push_back(get_bc_func()[ix2]);
    } else {  // block boundary
      op.bfuncs().push_back(nullptr);
    }

    // x2-outer
    auto ox2 = external_bc["x2-outer"].as<std::string>("reflecting");
    ox2 += "_outer";
    TORCH_CHECK(get_bc_func().find(ox2) != get_bc_func().end(),
                "Boundary function '", ox2, "' is not defined.");

    if (op.dist().lx2() == op.dist().nb2() - 1) {  // physical boundary
      op.bfuncs().push_back(get_bc_func()[ox2]);
    } else {  // block boundary
      op.bfuncs().push_back(nullptr);
    }

  } else if (op.hydro().coord().nc3() > 1) {
    op.bfuncs().push_back(nullptr);
    op.bfuncs().push_back(nullptr);
  }

  if (op.hydro().coord().nc3() > 1) {
    // x3-inner
    auto ix3 = external_bc["x3-inner"].as<std::string>("reflecting");
    ix3 += "_inner";
    TORCH_CHECK(get_bc_func().find(ix3) != get_bc_func().end(),
                "Boundary function '", ix3, "' is not defined.");

    if (op.dist().lx3() == 0) {  // physical boundary
      op.bfuncs().push_back(get_bc_func()[ix3]);
    } else {  // block boundary
      op.bfuncs().push_back(nullptr);
    }

    // x3-outer
    auto ox3 = external_bc["x3-outer"].as<std::string>("reflecting");
    ox3 += "_outer";
    TORCH_CHECK(get_bc_func().find(ox3) != get_bc_func().end(),
                "Boundary function '", ox3, "' is not defined.");

    if (op.dist().lx3() == op.dist().nb3() - 1) {  // physical boundary
      op.bfuncs().push_back(get_bc_func()[ox3]);
    } else {  // block boundary
      op.bfuncs().push_back(nullptr);
    }
  }

  // --------------- outputs --------------- //
  int fid = 0;
  if (config["outputs"]) {
    for (auto const& out_cfg : config["outputs"]) {
      op.outputs().push_back(OutputOptions::from_yaml(out_cfg, fid++));
    }
  }

  return op;
}

}  // namespace snap
