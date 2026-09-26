// C/C++
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <functional>
#include <string>

// external
#include <gtest/gtest.h>
#include <yaml-cpp/yaml.h>

// kintera
#include <kintera/constants.h>

// snap
#include <snap/snap.h>

#include <snap/coord/coordinate.hpp>
#include <snap/hydro/hydro.hpp>
#include <snap/mesh/meshblock.hpp>

using namespace snap;

namespace {
//! the one card this binary loads: kintera's species table is process-global
constexpr char const* kCard = "test_flux_positivity_carry.yaml";

struct Arm {
  std::shared_ptr<MeshBlockImpl> block;
  Variables vars;
  torch::Tensor du;
};

//! one hydro forward (dt = 1) of a uniform column, limiter on or off
Arm forward_once(bool limiter, std::function<void(YAML::Node&)> const& edit,
                 double vx, double vy, double vz = 0.) {
  auto card = YAML::LoadFile(kCard);
  card["dynamics"]["equation-of-state"]["limiter"] = limiter;
  edit(card);
  std::string name = std::string("test_flux_positivity_carry_") +
                     (limiter ? "on" : "off") + ".yaml";
  {
    std::ofstream(name) << card;
  }
  Arm arm;
  arm.block =
      std::make_shared<MeshBlockImpl>(MeshBlockOptionsImpl::from_yaml(name));
  std::remove(name.c_str());

  auto coord = arm.block->pcoord;
  auto w = torch::zeros({arm.block->phydro->peos->nvar(), coord->options->nc3(),
                         coord->options->nc2(), coord->options->nc1()},
                        torch::kFloat64);
  w[IDN].fill_(1.);
  w[IPR].fill_(1.e5);
  w[IVX].fill_(vx);
  w[IVY].fill_(vy);
  w[IVZ].fill_(vz);
  w[ICY].fill_(0.01);      // vapor
  w[ICY + 1].fill_(0.02);  // cloud
  arm.vars["hydro_w"] = w;
  arm.block->initialize(arm.vars);
  arm.du = arm.block->phydro->forward(1., arm.vars.at("hydro_u"), arm.vars);
  return arm;
}

//! Per face along x1 (dim 1, at the first interior x2 row) or x2 (dim 2, at
//! the first interior x1 cell): the limited arm's energy and momentum fluxes
//! must fall short of the unlimited ones by exactly what the withheld species
//! mass carries at its donor cell, and the column totals must not change.
void expect_carried(Arm const& off, Arm const& on, int dim = 1) {
  ASSERT_TRUE(torch::equal(off.vars.at("hydro_u"), on.vars.at("hydro_u")));
  auto peos = off.block->phydro->peos;
  auto pcoord = off.block->pcoord;
  int il = dim == 1 ? pcoord->il() : pcoord->jl();
  int iu = dim == 1 ? pcoord->iu() : pcoord->ju();
  int across = dim == 1 ? pcoord->jl() : pcoord->il();
  // row c of a (var, x3, x2, x1) tensor at position n along dim
  auto at = [&](torch::Tensor const& t, int c, int n) {
    return (dim == 1 ? t[c][0][across][n] : t[c][0][n][across]).item<double>();
  };
  auto w = off.vars.at("hydro_w");
  auto u = off.vars.at("hydro_u");
  auto temp = peos->compute("W->T", {w}).unsqueeze(0);
  // energy per unit species mass (internal + kinetic), from the settling
  // flux's own "W->E"; the vapour adds its partial pressure p_n / rho_n
  auto e = peos->compute("W->E", {w}) / (w[IDN] * w.narrow(0, ICY, 2));
  auto F0 = dim == 1 ? off.block->phydro->flux1() : off.block->phydro->flux2();
  auto F1 = dim == 1 ? on.block->phydro->flux1() : on.block->phydro->flux2();

  int limited = 0;
  for (int i = il; i <= iu + 1; ++i) {
    double dE = 0., dM[3] = {0., 0., 0.};
    for (int n = 0; n < 2; ++n) {
      double f0 = at(F0, ICY + n, i);
      double dm = f0 - at(F1, ICY + n, i);
      if (std::abs(dm) > 0.1 * std::abs(f0)) ++limited;
      int d = f0 > 0. ? i - 1 : i;  // the donor cell
      double h = at(e, n, d);
      if (n == 0) {
        h +=
            kintera::constants::Rgas / peos->species_weight(1) * at(temp, 0, d);
      }
      dE += dm * h;
      for (int k = 0; k < 3; ++k) {
        dM[k] += dm * at(u, IVX + k, d) / at(w, IDN, d);
      }
    }
    // the limiter must not touch the dry flux
    EXPECT_EQ(at(F0, IDN, i), at(F1, IDN, i)) << "dry mass flux, face " << i;
    double e0 = at(F0, IPR, i);
    double e1 = at(F1, IPR, i);
    EXPECT_NEAR(e0 - e1, dE, 1.e-12 * std::max(std::abs(e0), std::abs(dE)))
        << "energy flux, face " << i;
    for (int k = 0; k < 3; ++k) {
      double m0 = at(F0, IVX + k, i);
      double m1 = at(F1, IVX + k, i);
      EXPECT_NEAR(m0 - m1, dM[k],
                  1.e-12 * std::max(std::abs(m0), std::abs(dM[k])))
          << "momentum flux " << k << ", face " << i;
    }
  }
  EXPECT_GT(limited, 0) << "the limiter never withheld a species flux";

  // unit cell volumes: the correction only moves energy and momentum
  auto cells = off.block->part({0, 0, 0}, PartOptions().exterior(false));
  for (int c : {(int)IPR, (int)IVX, (int)IVY, (int)IVZ}) {
    auto d0 = off.du.index(cells)[c];
    double t0 = d0.sum().item<double>();
    double t1 = on.du.index(cells)[c].sum().item<double>();
    EXPECT_NEAR(t1, t0, 1.e-12 * d0.abs().sum().item<double>() + 1.e-12)
        << "column total of row " << c;
  }
}
}  // namespace

// A uniform column moving up at 2 m/s with dt = dx / 1: every face drains its
// lower cell of twice the vapour and cloud it holds, so theta = 1/2 there and
// half of each species flux is withheld (both Riemann solvers).
TEST(flux_positivity, withheld_advected_mass_keeps_its_energy_and_momentum) {
  for (std::string rs : {"lmars", "hllc"}) {
    SCOPED_TRACE(rs);
    auto edit = [&](YAML::Node& card) {
      card["dynamics"]["riemann-solver"]["type"] = rs;
    };
    auto off = forward_once(false, edit, 2., 3.);
    auto on = forward_once(true, edit, 2., 3.);
    expect_carried(off, on);
  }
}

// A resting column (x2 wind 3 m/s) whose cloud settles at 2 m/s: each face
// drains the cell above it of twice its cloud, theta = 1/2, and the withheld
// settling flux keeps its energy (no pressure share) and its x2 momentum.
TEST(flux_positivity, withheld_settling_mass_keeps_its_energy_and_momentum) {
  auto edit = [](YAML::Node& card) {
    card["sedimentation"] =
        YAML::Load("{radius: {}, density: {}, const-vsed: {cloud: -2.}}");
  };
  auto off = forward_once(false, edit, 0., 3.);
  auto on = forward_once(true, edit, 0., 3.);
  expect_carried(off, on);
}

// The advection case turned sideways: a six-by-six slab at rest in x1 moving
// at 2 m/s along x2 (3 m/s along x3), so the limited faces are x2 faces and
// the carry runs on the x2 flux.
TEST(flux_positivity, withheld_mass_keeps_its_energy_and_momentum_along_x2) {
  auto edit = [](YAML::Node& card) {
    card["geometry"]["bounds"]["x2max"] = 6.;
    card["geometry"]["cells"]["nx2"] = 6;
    card["boundary-condition"]["external"]["x2-inner"] = "reflecting";
    card["boundary-condition"]["external"]["x2-outer"] = "reflecting";
  };
  auto off = forward_once(false, edit, 0., 2., 3.);
  auto on = forward_once(true, edit, 0., 2., 3.);
  expect_carried(off, on, 2);
}
