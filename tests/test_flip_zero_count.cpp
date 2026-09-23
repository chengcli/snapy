// C/C++
#include <iostream>

// torch
#include <torch/torch.h>

// snap
#include <snap/bc/internal_boundary.hpp>

using namespace snap;

int main() {
  // the count is accumulated once per column, so many short columns make the
  // shared increment frequent relative to the work between increments
  auto mask = torch::zeros({96, 96, 16}, torch::kInt32);
  auto flat = mask.view(-1);
  flat.index_put_({torch::arange(0, flat.numel(), 37, torch::kLong)}, 1);

  auto op = InternalBoundaryOptionsImpl::create();
  op->max_iter() = 64;
  InternalBoundary pib(op);

  at::set_num_threads(1);
  int serial = 0;
  pib->rectify_solid(mask.clone(), serial);
  if (serial <= 0) {
    std::cerr << "fixture is inert: serial flip count is " << serial
              << std::endl;
    return 1;
  }

  at::set_num_threads(8);
  if (at::get_num_threads() < 2) {
    std::cerr << "parallel arm unavailable: at::get_num_threads() = "
              << at::get_num_threads() << std::endl;
    return 1;
  }

  for (int k = 0; k < 20; ++k) {
    int parallel = 0;
    pib->rectify_solid(mask.clone(), parallel);
    if (parallel != serial) {
      std::cerr << "flip count " << parallel << " on " << at::get_num_threads()
                << " threads != " << serial << " serial, repeat " << k
                << ": increments were lost" << std::endl;
      return 1;
    }
  }

  std::cout << "flip count " << serial << " on 1 and on "
            << at::get_num_threads() << " threads" << std::endl;
  return 0;
}
