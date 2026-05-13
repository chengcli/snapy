// external
#include <gtest/gtest.h>

// snap
#include <snap/output/netcdf_utils.hpp>

TEST(NetcdfUtils, keeps_valid_names) {
  EXPECT_EQ(snap::sanitize_netcdf_name("rho"), "rho");
  EXPECT_EQ(snap::sanitize_netcdf_name("theta_v"), "theta_v");
  EXPECT_EQ(snap::sanitize_netcdf_name("path_H2O"), "path_H2O");
}

TEST(NetcdfUtils, sanitizes_species_names) {
  EXPECT_EQ(snap::sanitize_netcdf_name("H2O(l)"), "H2O_l_");
  EXPECT_EQ(snap::sanitize_netcdf_name("H2O(l,p)"), "H2O_l_p_");
  EXPECT_EQ(snap::sanitize_netcdf_name("path_NH4SH(s,p)"), "path_NH4SH_s_p_");
}

TEST(NetcdfUtils, prefixes_invalid_leading_character) {
  EXPECT_EQ(snap::sanitize_netcdf_name("1bad"), "_1bad");
  EXPECT_EQ(snap::sanitize_netcdf_name(""), "_");
}
