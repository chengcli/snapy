// external
#include <gtest/gtest.h>

// snap
#include <snap/interface/athena_arrays.hpp>

namespace snap {
namespace {

TEST(AthenaArray, copy_assignment_allocates_destination) {
  AthenaArray<double> src;
  src.NewAthenaArray(2, 3);
  src(0, 0) = 1.0;
  src(0, 1) = 2.0;
  src(0, 2) = 3.0;
  src(1, 0) = 4.0;
  src(1, 1) = 5.0;
  src(1, 2) = 6.0;

  AthenaArray<double> dst;
  dst = src;

  EXPECT_EQ(dst.GetDim2(), 2);
  EXPECT_EQ(dst.GetDim1(), 3);
  EXPECT_EQ(dst(1, 2), 6.0);

  src(1, 2) = 99.0;
  EXPECT_EQ(dst(1, 2), 6.0);
}

TEST(AthenaArray, copy_assignment_reallocates_existing_destination) {
  AthenaArray<double> src;
  src.NewAthenaArray(2, 2, 2);
  for (int n = 0; n < src.GetDim3(); ++n) {
    for (int j = 0; j < src.GetDim2(); ++j) {
      for (int i = 0; i < src.GetDim1(); ++i) {
        src(n, j, i) = 10 * n + 2 * j + i;
      }
    }
  }

  AthenaArray<double> dst;
  dst.NewAthenaArray(5);
  dst(0) = -1.0;
  dst = src;

  EXPECT_EQ(dst.GetDim3(), 2);
  EXPECT_EQ(dst.GetDim2(), 2);
  EXPECT_EQ(dst.GetDim1(), 2);
  EXPECT_EQ(dst(1, 1, 1), 13.0);
}

TEST(AthenaArray, copy_from_shallow_slice_creates_owned_copy) {
  AthenaArray<double> src;
  src.NewAthenaArray(2, 3, 2, 2);
  for (int n = 0; n < src.GetDim4(); ++n) {
    for (int k = 0; k < src.GetDim3(); ++k) {
      for (int j = 0; j < src.GetDim2(); ++j) {
        for (int i = 0; i < src.GetDim1(); ++i) {
          src(n, k, j, i) = 100 * n + 10 * k + 2 * j + i;
        }
      }
    }
  }

  AthenaArray<double> slice;
  slice.InitWithShallowSlice(src, 4, 1, 1);

  AthenaArray<double> dst;
  dst = slice;

  EXPECT_EQ(dst.GetDim4(), 1);
  EXPECT_EQ(dst.GetDim3(), 3);
  EXPECT_EQ(dst.GetDim2(), 2);
  EXPECT_EQ(dst.GetDim1(), 2);
  EXPECT_EQ(dst(0, 1, 1, 1), 113.0);

  src(1, 1, 1, 1) = 999.0;
  EXPECT_EQ(dst(0, 1, 1, 1), 113.0);
}

}  // namespace
}  // namespace snap

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
