// external
#include <gtest/gtest.h>

// base
#include <globals.h>

// fvm
#include <fvm/coord/coordinate.hpp>

using namespace canoe;

TEST(Coordinate, constructor) {
  CoordinateOptions options;

  options.x1min(0)
      .x1max(1)
      .x2min(0)
      .x2max(1)
      .x3min(0)
      .x3max(1)
      .nx1(10)
      .nx2(4)
      .nx3(1);

  Cartesian pcoord = Cartesian(options);

  pcoord->pretty_print(std::cout);
}

int main(int argc, char **argv) {
  testing::InitGoogleTest(&argc, argv);
  start_logging(argc, argv);

  return RUN_ALL_TESTS();
}
