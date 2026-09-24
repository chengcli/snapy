// C/C++
#include <cstdio>
#include <fstream>
#include <string>

// external
#include <gtest/gtest.h>

// snap
#include <snap/hydro/hydro.hpp>

// Unknown keys under `dynamics:` must be rejected, not silently ignored: a
// typo'd or removed option would otherwise run as if applied, with nothing in
// the log. `positivity` is a real example -- a documented key that no code
// reads.
TEST(hydro_options, reject_unknown_dynamics_keys) {
  auto write = [](std::string const &fname, std::string const &extra) {
    std::ofstream f(fname);
    f << "dynamics:\n"
         "  equation-of-state:\n"
         "    type: ideal-gas\n"
         "  reconstruct:\n"
         "    vertical: {type: plm, scale: false, shock: false}\n"
         "    horizontal: {type: plm, scale: false, shock: false}\n"
         "  riemann-solver:\n"
         "    type: hllc\n"
         "  verbose: false\n"
      << extra;
  };
  std::string good = "test_hydro_options_good.yaml";
  std::string bad = "test_hydro_options_bad.yaml";

  write(good, "");
  EXPECT_NO_THROW(snap::HydroOptionsImpl::from_yaml(good));

  write(bad, "  positivity: true\n");
  EXPECT_ANY_THROW(snap::HydroOptionsImpl::from_yaml(bad));

  std::remove(good.c_str());
  std::remove(bad.c_str());
}
