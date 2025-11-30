// yaml
#include <yaml-cpp/yaml.h>

// snap
#include <snap/output/output_type.hpp>

#include "meshblock.hpp"

namespace snap {

MeshBlockOptions MeshBlockOptionsImpl::from_yaml(std::string input_file) {
  auto op = MeshBlockOptions::create();

  auto config = YAML::LoadFile(input_file);
  op->verbose() = config["verbose"].as<bool>(false);

  // distribution environment and layout
  if (config["distribute"]) {
    op->layout().type() =
        config["distribute"]["layout"].as<std::string>("slab");
    if (op->layout().type() == "slab") {
      op->layout().px(config["distribute"]["nb3"].as<int>(1));
      op->layout().py(config["distribute"]["nb2"].as<int>(1));
      op->layout().pz(config["distribute"]["nb1"].as<int>(1));

      TORCH_CHECK(
          op->layout().pz() == 1,
          "Slab layout only supports partitioning along x2-x3 directions.");
    } else {
      TORCH_CHECK(false, "Only 'slab' layout is supported currently.");
    }

    op->dist()->backend() =
        config["distribute"]["backend"].as<std::string>("gloo");
    op->dist()->verbose() = config["distribute"]["verbose"].as<bool>(false);
  }

  // use the basename of the input file as the basename of the output files
  op->basename() = input_file.substr(0, input_file.find_last_of('.'));

  op->hydro() = HydroOptions::from_yaml(input_file, op->layout());
  op->intg() = harp::IntegratorOptions::from_yaml(input_file);

  // --------------- boundary conditions --------------- //

  if (!config["boundary-condition"]) return op;
  if (!config["boundary-condition"]["external"]) return op;

  auto external_bc = config["boundary-condition"]["external"];

  if (op->hydro().coord().nc1() > 1) {
    // x1-inner
    auto ix1 = external_bc["x1-inner"].as<std::string>("reflecting");
    if (ix1 == "periodic") op->layout().periodic_z() = true;

    ix1 += "_inner";
    TORCH_CHECK(get_bc_func().find(ix1) != get_bc_func().end(),
                "Boundary function '", ix1, "' is not defined.");

    op->bfuncs().push_back(get_bc_func()[ix1]);

    // x1-outer
    auto ox1 = external_bc["x1-outer"].as<std::string>("reflecting");
    if (ox1 == "periodic") op->layout().periodic_z() = true;

    ox1 += "_outer";
    TORCH_CHECK(get_bc_func().find(ox1) != get_bc_func().end(),
                "Boundary function '", ox1, "' is not defined.");

    op->bfuncs().push_back(get_bc_func()[ox1]);
  } else if (op->hydro().coord().nc2() > 1 || op->hydro().coord().nc3() > 1) {
    op->bfuncs().push_back(nullptr);
    op->bfuncs().push_back(nullptr);
  }

  if (op->hydro().coord().nc2() > 1) {
    // x2-inner
    auto ix2 = external_bc["x2-inner"].as<std::string>("reflecting");
    if (ix2 == "periodic") op->layout().periodic_y() = true;

    ix2 += "_inner";
    TORCH_CHECK(get_bc_func().find(ix2) != get_bc_func().end(),
                "Boundary function '", ix2, "' is not defined.");

    op->bfuncs().push_back(get_bc_func()[ix2]);

    // x2-outer
    auto ox2 = external_bc["x2-outer"].as<std::string>("reflecting");
    if (ox2 == "periodic") op->layout().periodic_y() = true;

    ox2 += "_outer";
    TORCH_CHECK(get_bc_func().find(ox2) != get_bc_func().end(),
                "Boundary function '", ox2, "' is not defined.");

    op->bfuncs().push_back(get_bc_func()[ox2]);
  } else if (op->hydro().coord().nc3() > 1) {
    op->bfuncs().push_back(nullptr);
    op->bfuncs().push_back(nullptr);
  }

  if (op->hydro().coord().nc3() > 1) {
    // x3-inner
    auto ix3 = external_bc["x3-inner"].as<std::string>("reflecting");
    if (ix3 == "periodic") op->layout().periodic_x() = true;

    ix3 += "_inner";
    TORCH_CHECK(get_bc_func().find(ix3) != get_bc_func().end(),
                "Boundary function '", ix3, "' is not defined.");

    op->bfuncs().push_back(get_bc_func()[ix3]);

    // x3-outer
    auto ox3 = external_bc["x3-outer"].as<std::string>("reflecting");
    if (ox3 == "periodic") op->layout().periodic_x() = true;

    ox3 += "_outer";
    TORCH_CHECK(get_bc_func().find(ox3) != get_bc_func().end(),
                "Boundary function '", ox3, "' is not defined.");

    op->bfuncs().push_back(get_bc_func()[ox3]);
  }

  // --------------- outputs --------------- //
  int fid = 0;
  if (config["outputs"]) {
    for (auto const& out_cfg : config["outputs"]) {
      op->outputs().push_back(OutputOptions::from_yaml(out_cfg, fid++));
    }
  }

  return op;
}

}  // namespace snap
