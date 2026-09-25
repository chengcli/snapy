// C/C++
#include <cstdio>
#include <fstream>
#include <string>

// external
#include <gtest/gtest.h>

// snap
#include <snap/hydro/hydro.hpp>
#include <snap/mesh/meshblock.hpp>

// Unknown keys under `dynamics:` must be rejected, not silently ignored: a
// typo'd or removed option would otherwise run as if applied, with nothing in
// the log. `positivity` stands in for such a key: the parser does not accept
// it, and the error must name it.
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
  try {
    snap::HydroOptionsImpl::from_yaml(bad);
    FAIL() << "Expected unknown key 'dynamics/positivity' to throw";
  } catch (std::exception const &exc) {
    auto msg = std::string(exc.what());
    EXPECT_NE(msg.find("dynamics/positivity"), std::string::npos) << msg;
  }

  std::remove(good.c_str());
  std::remove(bad.c_str());
}

// a silently ignored fric-heat key would change a card's physics unlogged
TEST(hydro_options, reject_removed_and_unknown_forcing_keys) {
  auto write = [](std::string const &fname, std::string const &forcing) {
    std::ofstream f(fname);
    f << "dynamics:\n"
         "  equation-of-state:\n"
         "    type: ideal-gas\n"
         "forcing:\n"
      << forcing;
  };
  std::string good = "test_forcing_options_good.yaml";
  std::string removed = "test_forcing_options_removed.yaml";
  std::string unknown = "test_forcing_options_unknown.yaml";

  write(good, "  const-gravity: {grav1: -10.}\n");
  EXPECT_NO_THROW(snap::HydroOptionsImpl::from_yaml(good));

  auto refused_saying = [](std::string const &f, std::string const &needle) {
    try {
      snap::HydroOptionsImpl::from_yaml(f);
      return false;
    } catch (std::exception const &e) {
      return std::string(e.what()).find(needle) != std::string::npos;
    }
  };

  write(removed, "  fric-heat: {}\n");
  EXPECT_TRUE(refused_saying(removed, "has been removed"));

  write(unknown, "  no-such-forcing: {}\n");
  EXPECT_TRUE(refused_saying(unknown, "unknown key"));

  std::remove(good.c_str());
  std::remove(removed.c_str());
  std::remove(unknown.c_str());
}

// a scheme outside {0, 1, 9} was rejected only by the verbose report, while the
// x2/x3 acoustic CFL bound had already been dropped for it
TEST(hydro_options, reject_unsupported_implicit_scheme) {
  auto write = [](std::string const &fname, int scheme) {
    std::ofstream f(fname);
    f << "reference-state: {Tref: 300., Pref: 1.e5}\n"
         "species:\n"
         "  - name: dry\n"
         "    composition: {O: 0.42, N: 1.56, Ar: 0.01}\n"
         "    cv_R: 2.5\n"
         "geometry:\n"
         "  type: cartesian\n"
         "  bounds: {x1min: 0., x1max: 6., x2min: 0., x2max: 1., x3min: 0., "
         "x3max: 1.}\n"
         "  cells: {nx1: 6, nx2: 1, nx3: 1, nghost: 2}\n"
         "dynamics:\n"
         "  equation-of-state:\n"
         "    type: ideal-gas\n"
         "boundary-condition:\n"
         "  external: {x1-inner: reflecting, x1-outer: reflecting}\n"
         "integration:\n"
         "  type: rk3\n"
         "  implicit-scheme: "
      << scheme << "\n";
  };
  std::string f = "test_implicit_scheme.yaml";
  auto build = [&]() {
    std::make_shared<snap::MeshBlockImpl>(
        snap::MeshBlockOptionsImpl::from_yaml(f));
  };
  write(f, 1);
  EXPECT_NO_THROW(build());
  write(f, 3);
  try {
    build();
    ADD_FAILURE() << "unsupported implicit scheme accepted";
  } catch (std::exception const &e) {
    EXPECT_NE(std::string(e.what()).find("Unsupported implicit scheme"),
              std::string::npos)
        << e.what();
  }
  std::remove(f.c_str());
}
