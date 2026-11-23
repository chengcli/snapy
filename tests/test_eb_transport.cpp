#include <gtest/gtest.h>

#include <chrono>
#include <filesystem>
#include <snap/mesh/mesh.hpp>
#include <snap/output/output_type.hpp>

using namespace snap;

namespace {
Mesh mesh(int blocks, std::string integrator = "rk3",
          std::string output_dir = {}, bool periodic_x2 = false,
          torch::Device device = torch::kCPU) {
  auto b = MeshBlockOptionsImpl::from_yaml("test_eb_transport.yaml");
  b->intg()->type(integrator);
  if (!output_dir.empty()) {
    auto rst = OutputOptionsImpl::create();
    rst->file_type("restart");
    b->outputs({rst});
    b->output_dir(output_dir).basename("eb");
  }
  b->layout()->px(blocks).periodic_x(periodic_x2);
  if (device.is_cuda()) b->layout()->device("cuda").device_id(device.index());
  auto opts = MeshOptionsImpl::create();
  opts->block(b);
  opts->blocks_per_process(blocks);
  auto m = Mesh(opts);
  if (device.is_cuda()) m->to(device);
  return m;
}

MeshVariables initialize(Mesh &m, bool uniform) {
  MeshVariables vars;
  for (auto &b : m->blocks) {
    auto c = b->pcoord;
    auto nodes = torch::meshgrid({c->x3f, c->x2f, c->x1f}, "ij");
    // Small cut cells on both sides of the x2 decomposition seam.
    auto phi = nodes[2] + 0.3 * nodes[1] - 0.41;
    b->set_embedded_boundary(
        EBGeometry::from_level_set(phi, c->x1f, c->x2f, c->x3f));
    auto rho = uniform
                   ? torch::ones_like(
                         phi.slice(0, 0, -1).slice(1, 0, -1).slice(2, 0, -1))
                   : 1. + 0.05 * torch::sin(c->x2v.view({1, -1, 1}) * 6.) *
                              torch::cos(c->x1v * 7.);
    auto w = torch::zeros({5, c->x3v.size(0), c->x2v.size(0), c->x1v.size(0)},
                          rho.options());
    w[IDN].copy_(rho);
    w[IPR].copy_(rho);
    vars.push_back({{"hydro_w", w},
                    {"scalar_r", torch::full_like(w.narrow(0, 0, 1), 0.23)}});
  }
  m->initialize(vars);
  return vars;
}
}  // namespace

TEST(EmbeddedBoundary, single_block_periodic_face_metrics) {
  auto m = mesh(1, "rk3", {}, true);
  auto b = m->blocks[0];
  auto c = b->pcoord;
  auto nodes = torch::meshgrid({c->x3f, c->x2f, c->x1f}, "ij");
  auto make = [&](double slope) {
    return EBGeometry::from_level_set(nodes[2] - 0.41 + slope * nodes[1],
                                      c->x1f, c->x2f, c->x3f);
  };
  const auto hi = c->x2f.size(0) - 1 - c->options->nghost();
  auto g = make(0.);
  auto aperture = g.face_fraction[1].select(1, hi);
  auto tangent = g.face_centroid[1][0].select(1, hi);
  // Closed faces have no physical centroid. Matching open faces translate
  // by one period in x2 and have identical tangential coordinates.
  tangent.masked_fill_(aperture.eq(0), 17.);
  b->set_embedded_boundary(g);
  EXPECT_NO_THROW(b->prepare_embedded_boundary());

  tangent.add_((aperture.gt(0) & aperture.lt(1)) * 0.001);
  b->set_embedded_boundary(g);
  EXPECT_THROW(b->prepare_embedded_boundary(), c10::Error);

  // The sloped wall gives two different, locally consistent apertures.
  b->set_embedded_boundary(make(0.005));
  EXPECT_THROW(b->prepare_embedded_boundary(), c10::Error);
}

TEST(EmbeddedBoundary, moving_block_reinstalls_its_geometry) {
  auto m = mesh(1);
  auto vars = initialize(m, true);
  auto b = m->blocks.front();
  auto flags = b->eb_geometry()->flags.clone();
  b->to(vars.front().at("hydro_u").device());
  EXPECT_TRUE(torch::equal(flags, b->eb_geometry()->flags));
  EXPECT_NO_THROW(b->prepare_embedded_boundary());
}

TEST(EmbeddedBoundary, stationary_oblique_wall) {
  for (auto rk : {"rk1", "rk2", "rk3", "rk3s4"}) {
    auto m = mesh(1, rk);
    auto v = initialize(m, true);
    auto start = v[0]["hydro_u"].clone();
    auto dt = m->max_time_step(v);
    ASSERT_GT(dt, 0.);
    for (size_t stage = 0; stage < m->blocks[0]->pintg->stages.size(); ++stage)
      m->forward(v, dt, stage);
    EXPECT_TRUE(torch::allclose(v[0]["hydro_u"], start, 1.e-10, 1.e-10)) << rk;
  }
}

TEST(EmbeddedBoundary, decomposition) {
  auto single = mesh(1), split = mesh(2);
  auto a = initialize(single, false), b = initialize(split, false);
  double dt = single->max_time_step(a);
  for (size_t stage = 0; stage < single->blocks[0]->pintg->stages.size();
       ++stage) {
    single->forward(a, dt, stage);
    split->forward(b, dt, stage);
  }
  std::vector<torch::Tensor> parts;
  for (size_t i = 0; i < b.size(); ++i)
    parts.push_back(b[i]["hydro_u"].index(
        split->blocks[i]->part({0, 0, 0}, PartOptions().exterior(false))));
  auto full = a[0]["hydro_u"].index(
      single->blocks[0]->part({0, 0, 0}, PartOptions().exterior(false)));
  EXPECT_TRUE(torch::allclose(torch::cat(parts, 2), full, 1.e-10, 1.e-10));
}

static void check_restart_and_retry(torch::Device device) {
  auto dir = std::filesystem::temp_directory_path();
  dir /= "snapy-eb-" +
         std::to_string(
             std::chrono::steady_clock::now().time_since_epoch().count());
  std::filesystem::create_directories(dir);
  auto first = mesh(1, "rk3", dir.string(), false, device);
  auto v = initialize(first, false);
  ASSERT_EQ(v[0].at("hydro_u").device(), device);
  double dt = first->max_time_step(v);
  auto before = v[0]["hydro_u"].clone();
  first->forward(v, dt, 0);
  EXPECT_EQ(first->blocks[0]->apply_redo(v[0], 1), 1);
  EXPECT_TRUE(torch::equal(before, v[0]["hydro_u"]));
  first->make_outputs(v, 0.);
  auto next = mesh(1, "rk3", {}, false, device);
  MeshVariables restored(1);
  auto checkpoint = dir / "eb.00000.restart";
  next->initialize(restored, checkpoint.c_str());
  ASSERT_EQ(restored[0].at("hydro_u").device(), device);
  EXPECT_TRUE(torch::equal(v[0]["hydro_u"], restored[0]["hydro_u"]));
  auto mass = [&](MeshVariables const &state) {
    auto b = first->blocks[0];
    return (state[0].at("hydro_u")[IDN] * b->pcoord->cell_volume())
        .index(b->part({0, 0, 0}, PartOptions().exterior(false).ndim(3)))
        .sum()
        .item<double>();
  };
  double m0 = mass(v);
  for (size_t stage = 0; stage < first->blocks[0]->pintg->stages.size();
       ++stage) {
    first->forward(v, dt, stage);
    next->forward(restored, dt, stage);
  }
  EXPECT_TRUE(torch::equal(v[0]["hydro_u"], restored[0]["hydro_u"]));
  EXPECT_NEAR(mass(v), m0, 1.e-12);
  auto interior =
      first->blocks[0]->part({0, 0, 0}, PartOptions().exterior(false).ndim(3));
  auto fluid =
      first->blocks[0]->eb_geometry()->volume_fraction.index(interior).gt(0);
  auto r = v[0]["scalar_s"][0] / v[0]["hydro_u"][IDN];
  r = r.index(interior);
  EXPECT_TRUE(torch::allclose(r.masked_select(fluid),
                              torch::full_like(r.masked_select(fluid), 0.23)));
  // Some Linux LibTorch builds export a broken filesystem::remove_all.
  EXPECT_TRUE(std::filesystem::remove(checkpoint));
  EXPECT_TRUE(std::filesystem::remove(dir));
}

TEST(EmbeddedBoundary, restart_and_retry) {
  check_restart_and_retry(torch::kCPU);
}

TEST(EmbeddedBoundary, restart_and_retry_cuda) {
  if (!torch::cuda::is_available()) GTEST_SKIP() << "no CUDA device";
  check_restart_and_retry(torch::Device(torch::kCUDA, 0));
}

TEST(EmbeddedBoundary, rejects_retired_mask) {
  auto m = mesh(1);
  Variables v{{"solid", torch::zeros({1, 1, 1})}};
  EXPECT_THROW(m->blocks[0]->initialize_local(v), c10::Error);
}
