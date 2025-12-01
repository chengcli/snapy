// yaml
#include <yaml-cpp/yaml.h>

// snap
#include <snap/output/output_type.hpp>

#include "meshblock.hpp"

namespace snap {

MeshBlockOptions MeshBlockOptionsImpl::from_yaml(std::string input_file,
                                                 bool verbose) {
  auto op = MeshBlockOptionsImpl::create();

  op->layout() = LayoutOptionsImpl::from_yaml(input_file);

  if (verbose) {
    std::cout << "[MeshBlockOptions] layout options:" << std::endl;
    op->layout()->report(std::cout);
  }

  // use the basename of the input file as the basename of the output files
  op->basename() = input_file.substr(0, input_file.find_last_of('.'));

  if (verbose) {
    std::cout << "[MeshBlockOptions] basename = " << op->basename()
              << std::endl;
  }

  op->hydro() = HydroOptionsImpl::from_yaml(input_file, verbose);

  if (verbose) {
    std::cout << "[MeshBlockOptions] hydro options:" << std::endl;
    op->hydro()->report(std::cout);
  }

  op->scalar() = ScalarOptionsImpl::from_yaml(input_file, verbose);
  op->scalar()->coord() = op->hydro()->coord();
  op->scalar()->recon() = op->hydro()->recon23();

  if (verbose) {
    std::cout << "[MeshBlockOptions] scalar options:" << std::endl;
    op->scalar()->report(std::cout);
  }

  op->intg() = harp::IntegratorOptionsImpl::from_yaml(input_file);

  if (verbose) {
    std::cout << "[MeshBlockOptions] integrator options:" << std::endl;
    op->intg()->report(std::cout);
  }

  auto config = YAML::LoadFile(input_file);
  op->verbose() = config["verbose"].as<bool>(verbose);

  // --------------- outputs --------------- //
  int fid = 0;
  if (config["outputs"]) {
    for (auto const& out_cfg : config["outputs"]) {
      op->outputs().push_back(OutputOptionsImpl::from_yaml(out_cfg, fid++));
    }

    if (verbose) {
      std::cout << "[MeshBlockOptions] output options:" << std::endl;
      for (auto const& out_op : op->outputs()) {
        out_op->report(std::cout);
        std::cout << "------------------------" << std::endl;
      }
    }
  }

  // --------------- boundary conditions --------------- //

  if (!config["boundary-condition"]) return op;
  if (!config["boundary-condition"]["external"]) return op;

  auto external_bc = config["boundary-condition"]["external"];

  if (op->hydro()->coord()->nc1() > 1) {
    // x1-inner
    auto ix1 = external_bc["x1-inner"].as<std::string>("reflecting");
    if (ix1 == "periodic") op->layout()->periodic_z(true);

    ix1 += "_inner";
    TORCH_CHECK(get_bc_func().find(ix1) != get_bc_func().end(),
                "Boundary function '", ix1, "' is not defined.");

    op->bfuncs().push_back(get_bc_func()[ix1]);

    if (verbose) {
      std::cout << "[MeshBlockOptions] x1-inner BC: " << ix1 << std::endl;
    }

    // x1-outer
    auto ox1 = external_bc["x1-outer"].as<std::string>("reflecting");
    if (ox1 == "periodic") op->layout()->periodic_z(true);

    ox1 += "_outer";
    TORCH_CHECK(get_bc_func().find(ox1) != get_bc_func().end(),
                "Boundary function '", ox1, "' is not defined.");

    op->bfuncs().push_back(get_bc_func()[ox1]);

    if (verbose) {
      std::cout << "[MeshBlockOptions] x1-outer BC: " << ox1 << std::endl;
    }
  } else if (op->hydro()->coord()->nc2() > 1 ||
             op->hydro()->coord()->nc3() > 1) {
    op->bfuncs().push_back(nullptr);
    op->bfuncs().push_back(nullptr);
  }

  if (op->hydro()->coord()->nc2() > 1) {
    // x2-inner
    auto ix2 = external_bc["x2-inner"].as<std::string>("reflecting");
    if (ix2 == "periodic") op->layout()->periodic_y(true);

    ix2 += "_inner";
    TORCH_CHECK(get_bc_func().find(ix2) != get_bc_func().end(),
                "Boundary function '", ix2, "' is not defined.");

    op->bfuncs().push_back(get_bc_func()[ix2]);

    if (verbose) {
      std::cout << "[MeshBlockOptions] x2-inner BC: " << ix2 << std::endl;
    }

    // x2-outer
    auto ox2 = external_bc["x2-outer"].as<std::string>("reflecting");
    if (ox2 == "periodic") op->layout()->periodic_y(true);

    ox2 += "_outer";
    TORCH_CHECK(get_bc_func().find(ox2) != get_bc_func().end(),
                "Boundary function '", ox2, "' is not defined.");

    op->bfuncs().push_back(get_bc_func()[ox2]);

    if (verbose) {
      std::cout << "[MeshBlockOptions] x2-outer BC: " << ox2 << std::endl;
    }
  } else if (op->hydro()->coord()->nc3() > 1) {
    op->bfuncs().push_back(nullptr);
    op->bfuncs().push_back(nullptr);
  }

  if (op->hydro()->coord()->nc3() > 1) {
    // x3-inner
    auto ix3 = external_bc["x3-inner"].as<std::string>("reflecting");
    if (ix3 == "periodic") op->layout()->periodic_x(true);

    ix3 += "_inner";
    TORCH_CHECK(get_bc_func().find(ix3) != get_bc_func().end(),
                "Boundary function '", ix3, "' is not defined.");

    op->bfuncs().push_back(get_bc_func()[ix3]);

    if (verbose) {
      std::cout << "[MeshBlockOptions] x3-inner BC: " << ix3 << std::endl;
    }

    // x3-outer
    auto ox3 = external_bc["x3-outer"].as<std::string>("reflecting");
    if (ox3 == "periodic") op->layout()->periodic_x(true);

    ox3 += "_outer";
    TORCH_CHECK(get_bc_func().find(ox3) != get_bc_func().end(),
                "Boundary function '", ox3, "' is not defined.");

    op->bfuncs().push_back(get_bc_func()[ox3]);

    if (verbose) {
      std::cout << "[MeshBlockOptions] x3-outer BC: " << ox3 << std::endl;
    }
  }

  return op;
}

}  // namespace snap
