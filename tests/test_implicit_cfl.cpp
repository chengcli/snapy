#include <gtest/gtest.h>

#include <fstream>
#include <snap/implicit/implicit_hydro.hpp>
#include <string>

namespace {

void write_card(std::string const& path, std::string const& integration) {
  std::ofstream f(path);
  f << "integration:\n  implicit-scheme: 1\n" << integration;
}

}  // namespace

TEST(implicit_cfl, rejects_non_numeric_and_keeps_the_ranges) {
  for (char const* key : {"implicit-advection-cfl", "shear-cfl"}) {
    for (char const* bad : {"banana", "maybe", "~", "[]", ".nan", ".inf"}) {
      auto path = std::string("pr7-cfl-") + key + ".yaml";
      write_card(path, std::string("  ") + key + ": " + bad + "\n");
      EXPECT_THROW(snap::ImplicitOptionsImpl::from_yaml(path), c10::Error)
          << key << " " << bad;
      std::remove(path.c_str());
    }
  }
  {
    auto path = std::string("pr7-cfl-adv-neg.yaml");
    write_card(path, "  implicit-advection-cfl: -1\n");
    EXPECT_THROW(snap::ImplicitOptionsImpl::from_yaml(path), c10::Error);
    std::remove(path.c_str());
  }
  {
    auto path = std::string("pr7-cfl-shear-neg.yaml");
    write_card(path, "  shear-cfl: -1\n");
    EXPECT_THROW(snap::ImplicitOptionsImpl::from_yaml(path), c10::Error);
    std::remove(path.c_str());
  }
  {
    auto path = std::string("pr7-cfl-ok.yaml");
    write_card(path, "  implicit-advection-cfl: 2.5\n  shear-cfl: 0\n");
    auto op = snap::ImplicitOptionsImpl::from_yaml(path);
    ASSERT_TRUE(op);
    EXPECT_DOUBLE_EQ(op->advection_cfl(), 2.5);
    EXPECT_DOUBLE_EQ(op->shear_cfl(), 0.);
    std::remove(path.c_str());
  }
  {
    auto path = std::string("pr7-cfl-default.yaml");
    write_card(path, "");
    auto op = snap::ImplicitOptionsImpl::from_yaml(path);
    ASSERT_TRUE(op);
    EXPECT_DOUBLE_EQ(op->advection_cfl(), 1.);
    EXPECT_DOUBLE_EQ(op->shear_cfl(), 0.);
    std::remove(path.c_str());
  }
}
