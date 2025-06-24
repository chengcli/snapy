// C/C++
#include <chrono>
#include <future>
#include <stdexcept>

// torch
#include <torch/torch.h>

// base
#include <configure.h>

// snap
#include "mesh.hpp"
#include "mesh_formatter.hpp"
#include "meshblock.hpp"

namespace snap {
MeshImpl::MeshImpl(MeshOptions const& options_) : options(options_) { reset(); }

void MeshImpl::reset() {
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
        TORCH_CHECK(err == 0, "Error in block = {} with error = {}", i, err);
      } else {
        TORCH_CHECK(false, "Block = {} timed out", i);
      }
    }

    /*if (fatal_error_occurred.load()) {
      std::stringstream msg;
      msg << "FATAL ERROR occurred. Exiting..." << std::endl;
      throw std::runtime_error(msg.str());
    }*/

    time += dt;
    nstep++;
    if (max_steps > 0 && nstep >= max_steps) {
      // LOG_WARN(logger, "{} reached max_steps = {}", name(), max_steps);
      break;
    }

    break;
    load_balance();
  }
}

void MeshImpl::ApplyUserWorkBeforeOutput() {}

double MeshImpl::max_time_step() {
  double dt = 1.e9;
  for (auto block : blocks) {
    dt = std::min(dt, block->max_root_time_step(tree->root_level()));
  }

  return dt;
}

void MeshImpl::load_balance() {}
}  // namespace snap
