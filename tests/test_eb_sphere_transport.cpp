#include <gtest/gtest.h>

#include <cstdlib>
#include <snap/coord/cubed_sphere_utils.hpp>
#include <snap/eb/geometry.hpp>
#include <snap/eb/sphere_transport.hpp>
#include <snap/mesh/mesh.hpp>

namespace {
using namespace snap;

struct SphereRun {
  Mesh m;
  MeshVariables vars;
  std::vector<torch::Tensor> initial, initial_tracer;
  explicit SphereRun(int split, bool quiet, bool one_layer = false,
                     bool moving = false)
      : m(make(split, one_layer)) {
    for (auto& block : m->blocks) {
      auto c = block->pcoord;
      auto l = block->get_layout();
      int p = std::get<2>(l->loc_of(l->options->rank()));
      auto phi = torch::empty({c->x3f.numel(), c->x2f.numel(), c->x1f.numel()},
                              torch::kFloat64);
      auto values = phi.accessor<double, 3>();
      for (int k = 0; k < phi.size(0); ++k)
        for (int j = 0; j < phi.size(1); ++j) {
          auto unit = cs_ab_to_xyz(CS_FACE_NAMES[p], c->x2f[j].item<double>(),
                                   c->x3f[k].item<double>());
          for (int i = 0; i < phi.size(2); ++i)
            values[k][j][i] = unit.x - unit.y + 0.12 * unit.z - 0.075;
        }
      auto g = EBGeometry::from_cubed_sphere_level_set(phi, c->x1f, c->x2f,
                                                       c->x3f, p);
      block->set_embedded_boundary(g);
      auto w = torch::zeros_like(block->phydro->named_buffers()["D"]);
      w[IDN].fill_(1.);
      w[IPR].fill_(1.);
      if (!quiet) w[IPR] += 0.03 * g.cell_centroid[0] / 11.0;
      if (moving) {
        auto xyz = torch::zeros_like(w.narrow(0, IVX, 3));
        for (int k = 0; k < 3; ++k)
          xyz[k].copy_(0.01 * (k + 1) + 0.002 * g.cell_centroid[(k + 1) % 3]);
        // Keep total energy constant so every conserved row is affine.
        w[IPR] -= 0.5 * (block->phydro->peos->options->gammad() - 1) *
                  xyz.square().sum(0);
        w.narrow(0, IVX, 3).copy_(sphere_cart_to_velocity(*block, xyz));
      }
      vars.push_back({{"hydro_w", w},
                      {"scalar_r", torch::full_like(w.narrow(0, 0, 1), 0.17)}});
    }
    m->initialize(vars);
    for (auto& v : vars) {
      initial.push_back(v["hydro_u"].clone());
      initial_tracer.push_back(v["scalar_s"].clone());
    }
  }
  static Mesh make(int split, bool one_layer) {
    auto o = MeshOptionsImpl::from_yaml("test_eb_sphere_transport.yaml");
    o->block()->layout()->px(split).py(split);
    o->blocks_per_process(6 * split * split /
                          o->block()->layout()->process_world_size());
    if (one_layer) o->block()->coord()->global_nx1(1).nx1(1);
    return Mesh(o);
  }
  void advance(double dt) {
    for (size_t stage = 0; stage < m->blocks[0]->pintg->stages.size(); ++stage)
      m->forward(vars, dt, stage);
    ASSERT_EQ(m->check_redo(vars), 0);
  }
  torch::Tensor budgets(bool before) {
    auto total = torch::zeros({3}, torch::kFloat64);
    for (size_t i = 0; i < vars.size(); ++i) {
      auto b = m->blocks[i];
      auto active = b->part({0, 0, 0}, PartOptions().exterior(false));
      auto volume = b->pcoord->cell_volume().unsqueeze(0).index(active);
      auto u = (before ? initial[i] : vars[i].at("hydro_u")).index(active);
      auto tracer =
          (before ? initial_tracer[i] : vars[i].at("scalar_s")).index(active);
      total +=
          (torch::stack({u[IDN], u[IPR], tracer[0]}) * volume).sum({1, 2, 3});
    }
    return total;
  }
  torch::Tensor canonical(bool before) {
    auto c = m->blocks.front()->pcoord;
    auto out =
        torch::zeros({6, 6, c->options->global_nx3(), c->options->global_nx2(),
                      c->options->global_nx1()},
                     torch::kFloat64);
    for (size_t i = 0; i < vars.size(); ++i) {
      auto b = m->blocks[i];
      auto l = b->get_layout();
      auto [x, y, p] = l->loc_of(l->options->rank());
      auto v = before ? initial[i] : vars[i].at("hydro_u");
      auto u =
          torch::cat({v, before ? initial_tracer[i] : vars[i].at("scalar_s")});
      auto mom = u.narrow(0, IVX, 3);
      mom.copy_(sphere_momentum_to_cart(*b, mom));
      auto owned = u.index(b->part({0, 0, 0}, PartOptions().exterior(false)));
      int nb = owned.size(1), na = owned.size(2);
      out[p].narrow(1, y * nb, nb).narrow(2, x * na, na).copy_(owned);
    }
    auto layout = m->blocks.front()->get_layout();
    if (layout->has_process_group()) {
      std::vector<torch::Tensor> states{out};
      layout->comm->allreduce(states, c10d::ReduceOp::SUM);
    }
    return out;
  }
};
TEST(SphereEmbeddedTransport, requires_flux_in_each_resolved_direction) {
  for (bool one_layer : {false, true}) {
    auto mesh = SphereRun::make(1, one_layer);
    auto b = mesh->blocks.front();
    auto c = b->pcoord;
    auto phi = torch::ones({c->x3f.numel(), c->x2f.numel(), c->x1f.numel()},
                           c->x1f.options());
    auto g =
        EBGeometry::from_cubed_sphere_level_set(phi, c->x1f, c->x2f, c->x3f, 0);
    auto opt = b->phydro->options;
    for (int d = 1; d <= 3; ++d) {
      opt->disable_flux_x1(d == 1);
      opt->disable_flux_x2(d == 2);
      opt->disable_flux_x3(d == 3);
      if (one_layer && d == 1) {
        EXPECT_NO_THROW(b->set_embedded_boundary(g));
      } else {
        try {
          b->set_embedded_boundary(g);
          ADD_FAILURE() << "accepted disabled resolved direction x" << d;
        } catch (const c10::Error& e) {
          EXPECT_NE(
              std::string(e.what()).find(
                  "require flux in resolved direction x" + std::to_string(d)),
              std::string::npos);
        }
      }
    }
  }
}

TEST(SphereEmbeddedTransport, collapsed_radial_walls_resist_outward_flow) {
  SphereRun run(1, true, true);
  for (size_t i = 0; i < run.m->blocks.size(); ++i) {
    auto block = run.m->blocks[i];
    auto w = run.vars[i].at("hydro_w").clone();
    auto base = sphere_radial_wall_traction(*block->phydro, w);
    w[IVX] = 0.15;
    auto moving = sphere_radial_wall_traction(*block->phydro, w);
    auto radial = sphere_velocity_to_cart(*block, w.narrow(0, IVX, 3));
    auto active = block->part({0, 0, 0}, PartOptions().exterior(false));
    auto regular = (block->eb_geometry()->volume_fraction.eq(1))
                       .unsqueeze(0)
                       .index(active);
    auto work = ((moving - base) * radial)
                    .sum(0, true)
                    .index(active)
                    .masked_select(regular);
    if (work.numel()) EXPECT_GT(work.min().item<double>(), 0.);
  }
}

TEST(SphereEmbeddedTransport,
     initial_redistribution_preserves_affine_world_momentum) {
  SphereRun run(1, true, false, true);
  for (size_t i = 0; i < run.m->blocks.size(); ++i) {
    auto b = run.m->blocks[i];
    auto g = b->eb_geometry();
    auto active = b->part({0, 0, 0}, PartOptions().exterior(false));
    auto fluid = g->volume_fraction.unsqueeze(0).index(active)[0] > 0;
    auto p = sphere_momentum_to_cart(*b, run.initial[i].narrow(0, IVX, 3))
                 .index(active);
    auto x = g->cell_centroid.index(active);
    for (int d = 0; d < 3; ++d) {
      auto exact = 0.01 * (d + 1) + 0.002 * x[(d + 1) % 3];
      EXPECT_TRUE(torch::allclose(p[d].masked_select(fluid),
                                  exact.masked_select(fluid), 1e-11, 1e-12));
    }
  }
}

TEST(SphereEmbeddedTransport, radial_momentum_points_along_physical_panel_ray) {
  SphereRun run(1, true);
  for (auto& b : run.m->blocks) {
    auto c = b->pcoord;
    auto g = b->eb_geometry();
    auto radial = torch::zeros_like(g->cell_centroid);
    radial[0].fill_(1.);
    auto xyz = sphere_momentum_to_cart(*b, radial);
    for (int k : {c->kl(), c->ku()})
      for (int j : {c->jl(), c->ju()}) {
        auto ray =
            cs_ab_to_xyz(CS_FACE_NAMES[g->panel], c->x2v[j].item<double>(),
                         c->x3v[k].item<double>());
        auto value = xyz.index({torch::indexing::Slice(), k, j, c->il()});
        EXPECT_TRUE(torch::allclose(
            value, torch::tensor({ray.x, ray.y, ray.z}, value.options()), 1e-12,
            1e-12));
      }
  }
}

TEST(SphereEmbeddedTransport,
     rest_balances_open_faces_and_wall_in_world_frame) {
  for (bool one_layer : {false, true}) {
    SphereRun run(1, true, one_layer);
    run.advance(run.m->max_time_step(run.vars));
    EXPECT_TRUE(torch::allclose(run.canonical(false), run.canonical(true),
                                1e-10, 1e-10));
  }
}
TEST(SphereEmbeddedTransport, owner_graph_agrees_after_panel_subdivision) {
  const char* reference = std::getenv("SNAPY_SPHERE_REFERENCE");
  const char* world = std::getenv("WORLD_SIZE");
  if (world && std::stoi(world) > 1) {
    ASSERT_NE(reference, nullptr);
    SphereRun split(1, false);
    split.advance(split.m->max_time_step(split.vars));
    torch::Tensor expected;
    torch::load(expected, reference);
    EXPECT_TRUE(torch::allclose(split.canonical(false), expected, 1e-9, 1e-10));
    return;
  }
  SphereRun coarse(1, false), divided(2, false);
  auto before = coarse.canonical(true);
  auto dt = coarse.m->max_time_step(coarse.vars);
  coarse.advance(dt);
  divided.advance(dt);
  auto result = coarse.canonical(false);
  if (reference) torch::save(result, reference);
  EXPECT_TRUE(torch::allclose(coarse.budgets(false), coarse.budgets(true),
                              1e-12, 1e-12));
  EXPECT_TRUE(torch::allclose(divided.budgets(false), divided.budgets(true),
                              1e-12, 1e-12));
  EXPECT_GT((result - before).abs().max().item<double>(), 1e-5);
  EXPECT_TRUE(torch::allclose(result, divided.canonical(false), 1e-9, 1e-10));
  auto fluid = before.select(1, 5) > 0;
  EXPECT_TRUE(torch::allclose(result.select(1, 5).masked_select(fluid),
                              0.17 * result.select(1, 0).masked_select(fluid),
                              1e-10, 1e-10));
}
}  // namespace
