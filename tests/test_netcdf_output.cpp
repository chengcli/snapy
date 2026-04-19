// C/C++
#include <filesystem>
#include <fstream>
#include <string>

// external
#include <gtest/gtest.h>

#include "src/output/output_utils.hpp"

namespace fs = std::filesystem;

namespace {
fs::path make_test_root() {
  return fs::temp_directory_path() / "snapy_test_netcdf_output";
}
}  // namespace

TEST(NetcdfOutput, staging_path_uses_local_temp_directory) {
  auto final_path = fs::path("/tmp/snapy/output/example.nc");
  auto staged_path = snap::make_netcdf_staging_path(final_path);

  EXPECT_EQ(staged_path.parent_path(),
            fs::temp_directory_path() / "snapy-netcdf");
  EXPECT_NE(staged_path.filename().string().find(".example.nc.tmp."),
            std::string::npos);
}

TEST(NetcdfOutput, publish_staged_output_promotes_file_into_target_directory) {
  auto root = make_test_root();
  auto final_dir = root / "nested" / "nfs-like";
  auto final_path = final_dir / "result.nc";
  auto staged_path = snap::make_netcdf_staging_path(final_path);

  snap::ensure_output_directory(root.string());
  {
    std::ofstream out(staged_path);
    ASSERT_TRUE(out.is_open());
    out << "netcdf-placeholder";
  }

  ASSERT_TRUE(fs::exists(staged_path));
  EXPECT_NO_THROW(snap::publish_staged_output(staged_path, final_path));
  EXPECT_TRUE(fs::exists(final_path));
  EXPECT_FALSE(fs::exists(staged_path));

  std::ifstream in(final_path);
  std::string contents;
  std::getline(in, contents);
  EXPECT_EQ(contents, "netcdf-placeholder");
}

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
