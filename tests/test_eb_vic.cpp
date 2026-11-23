#include <gtest/gtest.h>

#include <snap/implicit/eb_wall.hpp>
#include <snap/mesh/meshblock.hpp>

using namespace snap;

TEST(EmbeddedVIC, reflected_wall_pressure_jacobian) {
  auto normal = torch::tensor({0.6, 0.8, 0.}, torch::kFloat64);
  auto gamma = torch::tensor(1.4, torch::kFloat64);
  for (double speed : {-0.2, 0.2}) {
    auto u = torch::tensor({1.2, speed, 0.04, -0.02, 2.6}, torch::kFloat64);
    auto state = [&](torch::Tensor z) {
      auto vel = z.narrow(0, IVX, 3) / z[IDN];
      auto p = (gamma - 1.) * (z[IPR] - .5 * z[IDN] * vel.square().sum());
      return torch::cat({z[IDN].reshape({1}), vel, p.reshape({1})});
    };
    for (bool hllc : {true, false}) {
      auto pressure = [&](torch::Tensor z) {
        auto w = state(z);
        auto y = (w[IDN] * gamma * w[IPR]).sqrt() *
                 (w.narrow(0, IVX, 3) * normal).sum();
        auto q = hllc ? (1. + (gamma + 1.) / (2. * gamma) *
                                  (y / w[IPR]).clamp_min(0.))
                            .sqrt()
                      : torch::ones_like(y);
        return (w[IPR] + q * y).item<double>();
      };
      auto grad = wall_pressure_gradient(state(u), state(u), normal, gamma,
                                         gamma, hllc);
      for (int j = 0; j < 5; ++j) {
        auto h = torch::zeros_like(u);
        h[j].fill_(1.e-6);
        EXPECT_NEAR(grad[j].item<double>(),
                    (pressure(u + h) - pressure(u - h)) / 2.e-6, 2.e-8);
      }
    }
  }
}

TEST(EmbeddedVIC, closed_segments_and_oblique_pressure) {
  for (int size : {1, 9})
    for (std::string solver : {"hllc", "lmars"})
      for (bool rest : {true, false}) {
        SCOPED_TRACE(solver);
        SCOPED_TRACE(size);
        SCOPED_TRACE(rest);
        auto op = MeshBlockOptionsImpl::from_yaml("test_eb_transport.yaml");
        op->hydro()->riemann()->type(solver);
        op->hydro()->icorr(ImplicitOptionsImpl::create());
        op->hydro()->icorr()->scheme(size);
        op->hydro()->icorr()->shear_cfl(.5);
        MeshBlock b(op);
        auto c = b->pcoord;
        auto nodes = torch::meshgrid({c->x3f, c->x2f, c->x1f}, "ij");
        // A tilted slab disconnects each vertical column into two fluid runs.
        auto coordinate = nodes[2] + .14 * nodes[1];
        auto phi = (coordinate - .39) * (coordinate - .62);
        auto g = EBGeometry::from_level_set(phi, c->x1f, c->x2f, c->x3f);
        b->set_embedded_boundary(g);
        auto rho = torch::ones_like(g.volume_fraction);
        auto w = torch::zeros({5, rho.size(0), rho.size(1), rho.size(2)},
                              rho.options());
        w[IDN].fill_(1.);
        w[IPR].copy_(rest ? rho : rho + .02 * torch::sin(6. * c->x1v));
        Variables vars{{"hydro_w", w},
                       {"scalar_r", torch::full_like(w.narrow(0, 0, 1), .2)}};
        b->initialize(vars);
        EXPECT_GT(b->local_max_time_step(vars), 0.);
        auto u0 = vars.at("hydro_u").clone();
        auto s0 = vars.at("scalar_s").clone();
        auto inside = b->part({0, 0, 0}, PartOptions().exterior(false));
        auto vol = c->cell_volume();
        // Vertical acoustic Courant number exceeds one.
        double dt = 1.3 * c->options->dx1();
        for (size_t k = 0; k < b->pintg->stages.size(); ++k)
          b->forward(vars, dt, k);
        auto u = vars.at("hydro_u");
        ASSERT_TRUE(u.isfinite().all().item<bool>());
        ASSERT_FALSE(b->phydro->picorr->solve_failed());
        if (rest)
          EXPECT_TRUE(torch::allclose(u.index(inside), u0.index(inside), 1.e-10,
                                      1.e-10));
        else
          EXPECT_GT((u - u0).index(inside).abs().max().item<double>(), 1.e-6);
        for (auto region :
             {g.cell_centroid[0] + .14 * g.cell_centroid[1] < .5,
              g.cell_centroid[0] + .14 * g.cell_centroid[1] >= .5}) {
          auto v = vol * region;
          auto delta = ((u - u0) * v).index(inside).sum({1, 2, 3});
          EXPECT_NEAR(delta[IDN].item<double>(), 0., 1.e-11);
          EXPECT_NEAR(delta[IPR].item<double>(), 0., 1.e-11);
          EXPECT_NEAR(((vars.at("scalar_s") - s0) * v)
                          .index(inside)
                          .sum()
                          .item<double>(),
                      0., 1.e-11);
        }
        auto& reg = b->phydro->flux_register();
        auto delta = ((u - u0) * vol).index(inside).sum({1, 2, 3});
        auto net =
            delta + reg.monitor().boundary(1) + reg.monitor().boundary(2);
        EXPECT_LT(net.abs().max().item<double>(), 1.e-9);
        auto moved = b->phydro->implicit_mass_correction()[IVX];
        auto closed = g.face_fraction[0].narrow(-1, 0, moved.size(-1)).eq(0.);
        EXPECT_EQ(moved.masked_select(closed).abs().max().item<double>(), 0.);
      }
}

TEST(EmbeddedVIC, moist_constituents_and_tracer) {
  for (std::string precision : {"float64", "float32"}) {
    SCOPED_TRACE(precision);
    auto op =
        MeshBlockOptionsImpl::from_yaml("test_tracer_dry_convention.yaml");
    op->hydro()->grav(nullptr);
    op->hydro()->icorr()->precision(precision);
    MeshBlock b(op);
    auto c = b->pcoord;
    auto nodes = torch::meshgrid({c->x3f, c->x2f, c->x1f}, "ij");
    auto x = nodes[2] / 32000.;
    auto g = EBGeometry::from_level_set((x - .356) * (x - .603), c->x1f, c->x2f,
                                        c->x3f);
    b->set_embedded_boundary(g);
    auto rho = torch::ones_like(g.volume_fraction);
    auto w =
        torch::zeros({7, rho.size(0), rho.size(1), rho.size(2)}, rho.options());
    w[IDN].fill_(.1);
    w[IPR].copy_(1.e5 * (1. + .01 * torch::cos(c->x1v * (6. / 32000.))));
    w[ICY].copy_(1.e-4 * (1. + .2 * torch::sin(c->x1v * (5. / 32000.))));
    Variables vars{{"hydro_w", w},
                   {"scalar_r", torch::full_like(w.narrow(0, 0, 1), .2)}};
    b->initialize(vars);
    auto u0 = vars.at("hydro_u").clone();
    auto inside = b->part({0, 0, 0}, PartOptions().exterior(false));
    double dt = .3;
    for (size_t k = 0; k < b->pintg->stages.size(); ++k)
      b->forward(vars, dt, k);
    ASSERT_FALSE(b->phydro->picorr->solve_failed());
    auto u = vars.at("hydro_u"), s = vars.at("scalar_s");
    ASSERT_TRUE(u.isfinite().all().item<bool>());
    double eps = precision == "float32" ? 1.e-6 : 1.e-10;
    auto difference =
        ((s - .2 * u.narrow(0, IDN, 1)) * (g.volume_fraction > 0.))
            .index(inside)
            .abs()
            .max()
            .item<double>();
    EXPECT_LT(difference, eps);
    for (auto region :
         {g.cell_centroid[0] < 16000., g.cell_centroid[0] >= 16000.}) {
      auto vol = c->cell_volume() * region;
      for (int row : {IDN, IPR, ICY}) {
        auto before =
            (u0.narrow(0, row, 1) * vol).index(inside).sum().item<double>();
        auto change = ((u - u0).narrow(0, row, 1) * vol)
                          .index(inside)
                          .sum()
                          .item<double>();
        EXPECT_NEAR(change / std::abs(before), 0., eps);
      }
    }
  }
}
