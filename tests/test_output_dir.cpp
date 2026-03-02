// C/C++
#include <filesystem>
#include <stdexcept>
#include <string>
#include <system_error>

// external
#include <gtest/gtest.h>

namespace fs = std::filesystem;

// Root of all temporary directories created by this test suite.
// Using temp_directory_path() keeps the tests cross-platform.
static const fs::path kTestRoot =
    fs::temp_directory_path() / "snapy_test_output";

// Mirrors the exact error-handling pattern used in
// src/output/netcdf.cpp and src/output/restart.cpp.
static void create_output_dir(const std::string &dir) {
  std::error_code ec;
  fs::create_directories(dir, ec);
  if (ec) {
    throw std::runtime_error("Failed to create output directory '" + dir +
                             "': " + ec.message());
  }
}

TEST(OutputDir, creates_missing_directory) {
  const fs::path dir = kTestRoot / "single";
  std::error_code ec;
  fs::remove_all(kTestRoot, ec);

  ASSERT_FALSE(fs::exists(dir));
  EXPECT_NO_THROW(create_output_dir(dir.string()));
  EXPECT_TRUE(fs::is_directory(dir));

  fs::remove_all(kTestRoot, ec);
}

TEST(OutputDir, creates_nested_missing_directory) {
  const fs::path dir = kTestRoot / "nested" / "deep" / "path";
  std::error_code ec;
  fs::remove_all(kTestRoot, ec);

  ASSERT_FALSE(fs::exists(dir));
  EXPECT_NO_THROW(create_output_dir(dir.string()));
  EXPECT_TRUE(fs::is_directory(dir));

  fs::remove_all(kTestRoot, ec);
}

TEST(OutputDir, idempotent_for_existing_directory) {
  const fs::path dir = kTestRoot / "existing";
  std::error_code ec;
  fs::create_directories(dir, ec);
  ASSERT_TRUE(fs::is_directory(dir));

  // calling again on an already-existing directory must not throw
  EXPECT_NO_THROW(create_output_dir(dir.string()));
  EXPECT_TRUE(fs::is_directory(dir));

  fs::remove_all(kTestRoot, ec);
}

TEST(OutputDir, current_directory_default) {
  // The default output_dir is "."; creating it must be a no-op, not an error.
  EXPECT_NO_THROW(create_output_dir("."));
}

int main(int argc, char **argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
