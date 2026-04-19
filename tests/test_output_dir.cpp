// C/C++
#include <atomic>
#include <chrono>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <system_error>

// external
#include <gtest/gtest.h>

#include "src/output/output_utils.hpp"

namespace fs = std::filesystem;

static fs::path make_unique_test_dir(const std::string& name) {
  static std::atomic<int> counter{0};
  fs::path path;
  do {
    auto now = std::chrono::steady_clock::now().time_since_epoch().count();
    auto suffix = counter.fetch_add(1, std::memory_order_relaxed);
    path = fs::temp_directory_path() /
           ("snapy_test_output_" + name + "_" + std::to_string(now) + "_" +
            std::to_string(suffix));
  } while (fs::exists(path));
  return path;
}

TEST(OutputDir, creates_missing_directory) {
  const fs::path dir = make_unique_test_dir("single");

  ASSERT_FALSE(fs::exists(dir));
  EXPECT_NO_THROW(snap::ensure_output_directory(dir.string()));
  EXPECT_TRUE(fs::is_directory(dir));
}

TEST(OutputDir, creates_nested_missing_directory) {
  const fs::path dir = make_unique_test_dir("nested") / "deep" / "path";

  ASSERT_FALSE(fs::exists(dir));
  EXPECT_NO_THROW(snap::ensure_output_directory(dir.string()));
  EXPECT_TRUE(fs::is_directory(dir));
}

TEST(OutputDir, idempotent_for_existing_directory) {
  const fs::path dir = make_unique_test_dir("existing");
  std::error_code ec;
  fs::create_directories(dir, ec);
  ASSERT_TRUE(fs::is_directory(dir));

  // calling again on an already-existing directory must not throw
  EXPECT_NO_THROW(snap::ensure_output_directory(dir.string()));
  EXPECT_TRUE(fs::is_directory(dir));
}

TEST(OutputDir, current_directory_default) {
  // The default output_dir is "."; creating it must be a no-op, not an error.
  EXPECT_NO_THROW(snap::ensure_output_directory("."));
}

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
