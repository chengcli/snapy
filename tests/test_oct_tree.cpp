// external
#include <gtest/gtest.h>

// torch
#include <torch/torch.h>

// base
#include <globals.h>

// fvm
#include <fvm/mesh/oct_tree.hpp>

using namespace canoe;

TEST(OctTreeTest, test1) {
  OctTree tree(OctTreeOptions().ndim(2).nb1(3).nb2(3).nb3(1));

  std::cout << tree->named_modules().size() << std::endl;

  for (const auto& m : tree->named_modules()) {
    // Print the name of each module
    std::cout << "Module name: " << m.key() << std::endl;
  }

  auto nodes = tree->forward();

  std::cout << "nodes.size(): " << nodes.size() << std::endl;

  for (const auto& node : nodes) {
    std::cout << "node: ";
    node->loc->pretty_print(std::cout);
    std::cout << std::endl;
  }

  // find neighbor
  LogicalLocation loc(2, 1, 0, 0);
  std::vector<BoundaryFlag> bcs = {
      BoundaryFlag::kPeriodic, BoundaryFlag::kPeriodic,
      BoundaryFlag::kPeriodic, BoundaryFlag::kPeriodic,
      BoundaryFlag::kPeriodic, BoundaryFlag::kPeriodic};
  auto nb = tree->find_neighbor(loc, {1, 0, 0}, bcs.data());
  if (nb.has_value()) {
    std::cout << "neighbor: ";
    nb.value()->loc->pretty_print(std::cout);
    std::cout << std::endl;
  } else {
    std::cout << "no neighbor" << std::endl;
  }
}

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  start_logging(argc, argv);

  return RUN_ALL_TESTS();
}
