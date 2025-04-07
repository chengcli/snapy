// C/C++
#include <chrono>
#include <future>
#include <stdexcept>

// spdlog
#include <configure.h>
#include <spdlog/sinks/basic_file_sink.h>

// base
#include <globals.h>

// fvm
#include "mesh.hpp"
#include "mesh_formatter.hpp"
#include "meshblock.hpp"

namespace snap {
MeshOptions::MeshOptions(ParameterInput pin) {
  x1min(pin->GetOrAddReal("mesh", "x1min", 0.0));
  x1max(pin->GetOrAddReal("mesh", "x1max", 1.0));

  x2min(pin->GetOrAddReal("mesh", "x2min", 0.0));
  x2max(pin->GetOrAddReal("mesh", "x2max", 1.0));

  x3min(pin->GetOrAddReal("mesh", "x3min", 0.0));
  x3max(pin->GetOrAddReal("mesh", "x3max", 1.0));

  ncycle(pin->GetOrAddInteger("mesh", "ncycle", 1));

  // inner-x1
  std::string bc = pin->GetOrAddString("mesh", "ix1_bc", "periodic");
  if (bc == "periodic") {
    bflags[0] = BoundaryFlag::kPeriodic;
  } else if (bc == "reflecting") {
    bflags[0] = BoundaryFlag::kReflect;
  } else if (bc == "outflow") {
    bflags[0] = BoundaryFlag::kOutflow;
  } else if (bc == "user") {
    bflags[0] = BoundaryFlag::kUser;
  } else {
    LOG_ERROR(logger, "Unknown boundary condition = {}", bc);
  }

  // outer-x1
  bc = pin->GetOrAddString("mesh", "ox1_bc", "periodic");
  if (bc == "periodic") {
    bflags[1] = BoundaryFlag::kPeriodic;
  } else if (bc == "reflecting") {
    bflags[1] = BoundaryFlag::kReflect;
  } else if (bc == "outflow") {
    bflags[1] = BoundaryFlag::kOutflow;
  } else if (bc == "user") {
    bflags[1] = BoundaryFlag::kUser;
  } else {
    LOG_ERROR(logger, "Unknown boundary condition = {}", bc);
  }

  // inner-x2
  bc = pin->GetOrAddString("mesh", "ix2_bc", "periodic");
  if (bc == "periodic") {
    bflags[2] = BoundaryFlag::kPeriodic;
  } else if (bc == "reflecting") {
    bflags[2] = BoundaryFlag::kReflect;
  } else if (bc == "outflow") {
    bflags[2] = BoundaryFlag::kOutflow;
  } else if (bc == "user") {
    bflags[2] = BoundaryFlag::kUser;
  } else {
    LOG_ERROR(logger, "Unknown boundary condition = {}", bc);
  }

  // outer-x2
  bc = pin->GetOrAddString("mesh", "ox2_bc", "periodic");
  if (bc == "periodic") {
    bflags[3] = BoundaryFlag::kPeriodic;
  } else if (bc == "reflecting") {
    bflags[3] = BoundaryFlag::kReflect;
  } else if (bc == "outflow") {
    bflags[3] = BoundaryFlag::kOutflow;
  } else if (bc == "user") {
    bflags[3] = BoundaryFlag::kUser;
  } else {
    LOG_ERROR(logger, "Unknown boundary condition = {}", bc);
  }

  // inner-x3
  bc = pin->GetOrAddString("mesh", "ix3_bc", "periodic");
  if (bc == "periodic") {
    bflags[4] = BoundaryFlag::kPeriodic;
  } else if (bc == "reflecting") {
    bflags[4] = BoundaryFlag::kReflect;
  } else if (bc == "outflow") {
    bflags[4] = BoundaryFlag::kOutflow;
  } else if (bc == "user") {
    bflags[4] = BoundaryFlag::kUser;
  } else {
    LOG_ERROR(logger, "Unknown boundary condition = {}", bc);
  }

  // outer-x3
  bc = pin->GetOrAddString("mesh", "ox3_bc", "periodic");
  if (bc == "periodic") {
    bflags[5] = BoundaryFlag::kPeriodic;
  } else if (bc == "reflecting") {
    bflags[5] = BoundaryFlag::kReflect;
  } else if (bc == "outflow") {
    bflags[5] = BoundaryFlag::kOutflow;
  } else if (bc == "user") {
    bflags[5] = BoundaryFlag::kUser;
  } else {
    LOG_ERROR(logger, "Unknown boundary condition = {}", bc);
  }

  block(MeshBlockOptions(pin));
  tree(OctTreeOptions(pin));
}

MeshImpl::MeshImpl(MeshOptions const& options_) : options(options_) { reset(); }

void MeshImpl::reset() {
  LOG_INFO(logger, "{} resets with options: {}", name(), options);

  tree = register_module("tree", OctTree(options.tree()));
  auto nodes = tree->forward();

  for (auto node : nodes) {
    auto pmb = MeshBlock(options.block(), node->loc);
    blocks.push_back(pmb);
  }

  for (auto i = 0; i < blocks.size(); i++) {
    blocks[i]->initialize(options, tree);
    register_module("block" + std::to_string(i), blocks[i]);
  }
}

void MeshImpl::forward(double time, int max_steps) {
  LOG_INFO(logger, "{} will march to time = {}", name(), time);

  int nstep = 0;
  while (time > current_time) {
    auto dt = max_time_step();
    if (time - current_time < dt) {
      dt = time - current_time;
    }

    std::vector<std::future<int>> jobs;

    for (auto pmb : blocks) {
      jobs.push_back(std::async(std::launch::async, [&]() -> int {
        for (int stage = 0; stage < pmb->pintg->stages.size(); stage++) {
          auto err = pmb->forward(dt, stage);
          if (err != 0) {
            return err;
          }
        }
        return 0;
      }));
    }

    for (int i = 0; i < jobs.size(); i++) {
      if (jobs[i].wait_for(std::chrono::seconds(timeout_)) ==
          std::future_status::ready) {
        auto err = jobs[i].get();
        if (err != 0) {
          LOG_ERROR(logger, "{} failed at block = {} with error = {}", name(),
                    i, err);
        }
      } else {
        LOG_ERROR(logger, "{} timed out at block = {}", name(), i);
      }
    }

    if (fatal_error_occurred.load()) {
      std::stringstream msg;
      msg << "FATAL ERROR occurred. Exiting..." << std::endl;
      throw std::runtime_error(msg.str());
    }

    time += dt;
    nstep++;
    LOG_INFO(logger, "cycle #{}, {} marched to time = {}", nstep, name(),
             current_time);

    if (max_steps > 0 && nstep >= max_steps) {
      LOG_WARN(logger, "{} reached max_steps = {}", name(), max_steps);
      break;
    }

    break;
    load_balance();
  }
}

void MeshImpl::ApplyUserWorkBeforeOutput() {}

double MeshImpl::max_time_step() {
  LOG_INFO(logger, "{} calculates max time step", name());

  double dt = 1.e9;
  for (auto block : blocks) {
    dt = std::min(dt, block->max_root_time_step(tree->root_level()));
  }

  return dt;
}

void MeshImpl::load_balance() {
  LOG_INFO(logger, "{} does load balance", name());
}
}  // namespace snap
