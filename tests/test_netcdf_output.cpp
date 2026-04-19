// C/C++
#include <cstdlib>
#include <string>

// external
#include <gtest/gtest.h>

#include "src/output/output_utils.hpp"

TEST(NetcdfOutput, configure_hdf5_file_locking_sets_default_once) {
#if defined(_WIN32)
  _putenv_s("HDF5_USE_FILE_LOCKING", "");
#else
  unsetenv("HDF5_USE_FILE_LOCKING");
#endif

  snap::configure_hdf5_file_locking_for_netcdf();

  auto* value = std::getenv("HDF5_USE_FILE_LOCKING");
  ASSERT_NE(value, nullptr);
  EXPECT_STREQ(value, "FALSE");
}

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
