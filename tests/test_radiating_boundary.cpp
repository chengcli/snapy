#include <gtest/gtest.h>

#include <cmath>
#include <kintera/utils/serialize.hpp>
#include <limits>
#include <snap/mesh/mesh.hpp>
#include <snap/mesh/meshblock.hpp>

#include "device_testing.hpp"

using namespace snap;
namespace {
MeshBlock block_for(std::string eos = "ideal-gas", int nx = 8,
                    bool line = false, int ntracer = 0) {
  auto op = MeshBlockOptionsImpl::from_yaml(eos == "ideal-gas"
                                                ? "test_radiating_boundary.yaml"
                                                : "test_radiating_moist.yaml");
  op->hydro()->eos()->type(eos);
  op->scalar()->nvar(ntracer);
  op->coord()->nx1(nx).global_nx1(nx);
  if (line) op->coord()->nx2(1).nx3(1).global_nx2(1).global_nx3(1);
  return MeshBlock(op);
}
torch::Tensor background(MeshBlock b) {
  auto c = b->pcoord->options;
  auto w = torch::zeros({b->phydro->peos->nvar(), c->nc3(), c->nc2(), c->nc1()},
                        b->pcoord->x1v.options());
  w[IDN].fill_(1.);
  w[IPR].fill_(100000.);
  if (w.size(0) > 5) w[ICY].fill_(0.1);
  return w;
}
BoundaryFuncOptions context(MeshBlock b, torch::Tensor ref) {
  BoundaryFuncOptions op;
  op.type(kPrimitive).nghost(3);
  op.eos = b->phydro->peos.get();
  op.coord = b->pcoord.get();
  op.reference = ref.clone();
  return op;
}
}  // namespace

TEST_P(DeviceTest, radiating_modes_all_faces) {
  for (auto eos : {"ideal-gas", "ideal-moist", "moist-mixture"}) {
    auto b = block_for(eos);
    b->to(device, dtype);
    for (int axis = 1; axis <= 3; ++axis)
      for (bool outer : {false, true}) {
        int dim = 4 - axis, vn = IVX + axis - 1;
        double sign = outer ? 1. : -1.;
        auto ref = background(b);
        auto cs = b->phydro->peos->compute(
            "WA->L", {ref, b->phydro->peos->compute("W->A", {ref})});
        double c = cs.flatten()[0].item<double>();
        for (double mach : {-2., -1., -0.3, 0., 0.3, 1., 2.}) {
          SCOPED_TRACE(std::string(eos) + " axis=" + std::to_string(axis) +
                       " outer=" + std::to_string(outer) +
                       " M=" + std::to_string(mach));
          // Keep rho,p and current c unchanged so sonic comparisons are exact.
          ref[vn].fill_(sign * (mach * c - 2.));
          auto w = ref.clone();
          int src = outer ? w.size(dim) - 4 : 3,
              dst = outer ? w.size(dim) - 3 : 0;
          auto wi = w.narrow(dim, src, 1);
          wi[vn].fill_(sign * mach * c);
          int vt = IVX + axis % 3;
          wi[vt].add_(3.);
          auto before = w.clone();
          auto op = context(b, ref);
          op.tracer_reference = torch::full(
              {1, w.size(1), w.size(2), w.size(3)}, 0.2, w.options());
          op.tracers = op.tracer_reference.clone();
          op.tracers.narrow(dim, src, 1).add_(0.1);
          auto fn = get_bc_func().at(outer ? "outflow_outer" : "outflow_inner");
          fn(w, dim, op);
          auto g = w.narrow(dim, dst, 3) - ref.narrow(dim, dst, 3);
          double plus = mach >= -1 ? 2. : 0., minus = mach >= 1 ? 2. : 0.;
          double tol = dtype == torch::kFloat32 ? 0.03 : 1.e-8;
          EXPECT_LT(
              (g[vn] - sign * 0.5 * (plus + minus)).abs().max().item<double>(),
              tol);
          EXPECT_LT(
              (g[IPR] - 0.5 * c * (plus - minus)).abs().max().item<double>(),
              tol);
          EXPECT_LT((g[vt] - (mach >= 0 ? 3. : 0.)).abs().max().item<double>(),
                    tol);
          EXPECT_TRUE(
              torch::equal(w.narrow(dim, 3, 8), before.narrow(dim, 3, 8)));
          EXPECT_TRUE(
              torch::allclose(op.tracers.narrow(dim, dst, 3),
                              torch::full_like(op.tracers.narrow(dim, dst, 3),
                                               mach >= 0 ? 0.3 : 0.2)));
        }
      }
  }
}

TEST_P(DeviceTest, radiating_background_and_coordinate_roundtrip) {
  auto b = block_for();
  b->to(device, dtype);
  // Exercise a spatially varying, nonorthogonal g23 metric.
  b->pcoord->cosine_cell_kj.copy_(
      torch::linspace(-0.6, 0.6, 14, b->pcoord->x1v.options())
          .view({1, 14, 1})
          .expand({14, 14, 1}));
  auto ref = background(b);
  ref[IDN].mul_(torch::linspace(0.8, 1.2, 14, ref.options()));
  ref[IPR].mul_(torch::linspace(1.3, 0.7, 14, ref.options()));
  ref.narrow(0, IVX, 3).normal_();
  for (int axis = 1; axis <= 3; ++axis) {
    auto round = ref.clone();
    b->pcoord->boundary_velocity_(round, axis);
    b->pcoord->boundary_velocity_(round, axis, true);
    EXPECT_TRUE(torch::allclose(round, ref, 1.e-5, 1.e-6));
    for (bool outer : {false, true}) {
      auto w = ref.clone();
      get_bc_func().at(outer ? "outflow_outer" : "outflow_inner")(
          w, 4 - axis, context(b, ref));
      EXPECT_TRUE(torch::allclose(w, ref, 1.e-5, 1.e-6));
    }
  }
}

TEST(radiating, admissibility_and_errors) {
  auto b = block_for("ideal-moist");
  auto ref = background(b), w = ref.clone();
  // Valid interior but a near-vacuum ghost background: outgoing rarefaction
  // must be reduced, with velocity and tracers using the same factor.
  ref[IPR].narrow(2, 0, 3).fill_(1.);
  w = ref.clone();
  w[IVX].narrow(2, 3, 1).fill_(100.);
  auto op = context(b, ref);
  get_bc_func().at("outflow_inner")(w, 3, op);
  EXPECT_GT(w[IPR].min().item<double>(), 0.);
  EXPECT_LT(w[IVX].narrow(2, 0, 3).abs().max().item<double>(), 100.);
  ref[IDN].zero_();
  EXPECT_THROW(get_bc_func().at("outflow_inner")(w, 3, context(b, ref)),
               c10::Error);
  ref = background(b);
  w[IVX].fill_(std::numeric_limits<double>::quiet_NaN());
  EXPECT_THROW(get_bc_func().at("outflow_inner")(w, 3, context(b, ref)),
               c10::Error);
  auto aux = torch::rand({1, 1, 1, 14}, torch::kFloat64);
  auto expected = aux.narrow(3, 3, 1).clone();
  BoundaryFuncOptions scalar;
  scalar.type(kScalar).nghost(3);
  get_bc_func().at("outflow_inner")(aux, 3, scalar);
  EXPECT_TRUE(torch::equal(aux.narrow(3, 0, 3), expected.expand({1, 1, 1, 3})));
}

TEST(radiating, reference_copies_and_active_conserved_unchanged) {
  auto b = block_for("ideal-gas", 8, true);
  Variables vars{{"hydro_w", background(b)}};
  b->initialize(vars);
  auto ref = vars.at("boundary_reference_w").clone();
  vars.at("hydro_w")[IVX].fill_(2.);
  EXPECT_TRUE(torch::equal(ref, vars.at("boundary_reference_w")));
  auto u = vars.at("hydro_u");
  u[IVX].narrow(2, 3, 8).add_(0.1);
  auto active = u.narrow(3, 3, 8).clone();
  b->apply_boundaries(vars, u);
  EXPECT_TRUE(torch::equal(active, u.narrow(3, 3, 8)));
}

namespace {
void step(MeshBlock b, Variables& v, double dt) {
  ++b->cycle;
  for (int stage = 0; stage < b->pintg->stages.size(); ++stage)
    b->forward(v, dt, stage);
}
void save_restart(Variables const& v, int cycle, std::string path) {
  Variables data;
  for (auto const& [k, t] : v) data[k] = t.clone();
  data["last_time"] = torch::tensor({0.}, torch::kFloat64);
  data["last_cycle"] = torch::tensor({int64_t(cycle)}, torch::kInt64);
  data["file_number"] = torch::empty({0}, torch::kInt64);
  data["next_time"] = torch::empty({0}, torch::kFloat64);
  kintera::save_tensors(data, path);
}
}  // namespace

TEST(radiating, acoustic_pulses) {
  constexpr int nx = 128;
  constexpr double eps = 1.e-4;
  double c = std::sqrt(1.4);
  for (auto scheme : {"plm", "weno5"})
    for (double sign : {-1., 1.}) {
      for (auto bc : {"outflow", "extrapolation"}) {
        auto op =
            MeshBlockOptionsImpl::from_yaml("test_radiating_boundary.yaml");
        op->hydro()->eos()->type("ideal-gas");
        op->hydro()->recon1()->interp()->type(scheme);
        op->coord()
            ->nx1(nx)
            .global_nx1(nx)
            .nx2(1)
            .nx3(1)
            .global_nx2(1)
            .global_nx3(1);
        op->bfuncs()[0] = get_bc_func().at(std::string(bc) + "_inner");
        op->bfuncs()[1] = get_bc_func().at(std::string(bc) + "_outer");
        auto b = MeshBlock(op);
        auto w = background(b);
        w[IPR].fill_(1.);
        Variables v{{"hydro_w", w}};
        b->initialize(v);
        // Launch after capturing the fixed uniform reference.
        auto pulse =
            eps * torch::exp(-torch::square((b->pcoord->x1v - 0.5) / 0.06));
        w[IDN].add_(pulse);
        w[IPR].add_(c * c * pulse);
        w[IVX].add_(sign * c * pulse);
        v["hydro_u"] = b->phydro->peos->compute("W->U", {w});
        double time = 0., end = 0.85 / c;
        while (time < end) {
          double dt = std::min(0.3 / (nx * c), end - time);
          step(b, v, dt);
          time += dt;
        }
        auto result = b->phydro->peos->compute("U->W", {v.at("hydro_u")});
        auto incoming = sign * result[IVX] - (result[IPR] - 1.) / c;
        double reflection =
            incoming.narrow(2, 3, nx).abs().max().item<double>() /
            (2 * c * eps);
        std::cout << "pulse " << scheme << " sign=" << sign << " " << bc
                  << " reflected fraction=" << reflection << std::endl;
        if (std::string(bc) == "outflow") EXPECT_LT(reflection, 0.05);
      }
    }
}

TEST(radiating, restart_continuation_and_missing_reference) {
  auto b = block_for("ideal-gas", 24, true, 1);
  Variables v{{"hydro_w", background(b)},
              {"scalar_r", torch::full({1, 1, 1, 30}, 0.2, torch::kFloat64)}};
  b->initialize(v);
  v.at("hydro_u")[IVX].narrow(2, 3, 24).fill_(0.1);
  double dt = 1.e-5;
  step(b, v, dt);
  std::string path = "/tmp/snapy-radiating-restart.pt";
  save_restart(v, b->cycle, path);
  auto restarted = block_for("ideal-gas", 24, true, 1);
  Variables r;
  restarted->initialize(r, path.c_str());
  EXPECT_TRUE(
      torch::equal(v.at("boundary_reference_w"), r.at("boundary_reference_w")));
  for (int n = 0; n < 3; ++n) {
    step(b, v, dt);
    step(restarted, r, dt);
  }
  EXPECT_TRUE(torch::equal(v.at("hydro_u"), r.at("hydro_u")));
  EXPECT_TRUE(torch::equal(v.at("scalar_s"), r.at("scalar_s")));
  v.erase("boundary_reference_w");
  save_restart(v, b->cycle, path);
  EXPECT_THROW(restarted->initialize(r, path.c_str()), c10::Error);
  std::remove(path.c_str());
}

TEST(radiating, decomposed_physical_faces_match) {
  auto make_mesh = [](int count) {
    auto op = MeshOptionsImpl::create();
    op->block(MeshBlockOptionsImpl::from_yaml("test_radiating_boundary.yaml"));
    auto c = op->block()->coord();
    c->nx1(24).global_nx1(24).nx2(1).global_nx2(1).nx3(1).global_nx3(1);
    op->block()->hydro()->eos()->type("ideal-gas");
    op->block()->layout()->type("cubed").pz(count);
    op->blocks_per_process(count);
    return Mesh(op);
  };
  auto single = make_mesh(1), split = make_mesh(2);
  MeshVariables one(1), two(2);
  one[0]["hydro_w"] = background(single->blocks[0]);
  for (int n = 0; n < 2; ++n) two[n]["hydro_w"] = background(split->blocks[n]);
  single->initialize(one);
  split->initialize(two);
  auto perturb = [](Mesh mesh, MeshVariables& vars) {
    for (int n = 0; n < mesh->blocks.size(); ++n) {
      auto b = mesh->blocks[n];
      auto w = vars[n].at("hydro_w");
      w[IVX].copy_(torch::sin(6.28 * b->pcoord->x1v).view({1, 1, -1}));
      vars[n]["hydro_u"] = b->phydro->peos->compute("W->U", {w});
      b->apply_boundaries(vars[n], vars[n].at("hydro_u"));
    }
  };
  perturb(single, one);
  perturb(split, two);
  EXPECT_EQ(split->blocks[0]->options->bfuncs()[1], nullptr);
  EXPECT_EQ(split->blocks[1]->options->bfuncs()[0], nullptr);
  for (int n = 0; n < 2; ++n) {
    auto u = two[n].at("hydro_u");
    EXPECT_TRUE(torch::allclose(u, one[0].at("hydro_u").narrow(3, n * 12, 18),
                                1.e-12, 1.e-12));
  }
  // Advance through actual local halo exchanges and compare physical faces.
  for (int stage = 0; stage < single->blocks[0]->pintg->stages.size();
       ++stage) {
    single->forward(one, 1.e-5, stage);
    split->forward(two, 1.e-5, stage);
  }
  EXPECT_TRUE(torch::allclose(two[0].at("hydro_u").narrow(3, 0, 3),
                              one[0].at("hydro_u").narrow(3, 0, 3), 1.e-12,
                              1.e-12));
  EXPECT_TRUE(torch::allclose(two[1].at("hydro_u").narrow(3, 15, 3),
                              one[0].at("hydro_u").narrow(3, 27, 3), 1.e-12,
                              1.e-12));
}

TEST(radiating, shallow_water_and_unsupported_eos) {
  auto b = block_for();
  auto ref = background(b), w = ref.clone();
  b->phydro->peos->options->type("unsupported");
  EXPECT_THROW(get_bc_func().at("outflow_inner")(w, 3, context(b, ref)),
               c10::Error);
  b->phydro->peos->options->type("shallow-water");
  w = torch::rand({4, 1, 1, 14}, torch::kFloat64);
  auto expected = w.narrow(3, 3, 1).clone();
  auto op = context(b, ref);
  op.type(kConserved);
  get_bc_func().at("outflow_inner")(w, 3, op);
  EXPECT_TRUE(torch::equal(w.narrow(3, 0, 3), expected.expand({4, 1, 1, 3})));
}

TEST(radiating, composition_entropy_and_joint_limiter) {
  auto b = block_for("ideal-moist", 8, true);
  auto ref = background(b);
  ref[IVX].fill_(-1000.);
  ref[ICY].narrow(2, 0, 3).fill_(0.9);
  auto w = ref.clone();
  w[ICY].narrow(2, 3, 1).fill_(0.4);
  w[IDN].narrow(2, 3, 1).add_(0.2);
  w[IVX].narrow(2, 3, 1).sub_(10.);
  auto op = context(b, ref);
  op.tracer_reference = torch::full({1, 1, 1, 14}, 0.2, torch::kFloat64);
  op.tracers = op.tracer_reference.clone();
  op.tracers.narrow(3, 3, 1).add_(0.1);
  get_bc_func().at("outflow_inner")(w, 3, op);
  auto alpha = (w[ICY].narrow(2, 0, 3) - 0.9) / 0.3;
  EXPECT_TRUE((alpha > 0).all().item<bool>());
  EXPECT_TRUE((alpha < 1).all().item<bool>());
  EXPECT_TRUE((w[ICY] <= 1).all().item<bool>());
  EXPECT_TRUE(torch::allclose(w[IDN].narrow(2, 0, 3), 1. + 0.2 * alpha));
  EXPECT_TRUE(torch::allclose(w[IVX].narrow(2, 0, 3), -1000. - 10. * alpha));
  EXPECT_TRUE(
      torch::allclose(op.tracers[0].narrow(2, 0, 3), 0.2 + 0.1 * alpha));
  ref[ICY].fill_(1.1);
  EXPECT_THROW(get_bc_func().at("outflow_inner")(w, 3, context(b, ref)),
               c10::Error);
}

TEST(radiating, cpu_cuda_agreement) {
  if (!torch::cuda::is_available()) GTEST_SKIP() << "CUDA unavailable";
  auto b = block_for("moist-mixture");
  auto ref = background(b), input = ref.clone();
  input[IVX].fill_(2.);
  input[IVY].fill_(-1.);
  input[IVZ].fill_(0.5);
  input[IPR].add_(100.);
  auto cpu = input.clone();
  for (int f = 0; f < 6; ++f)
    b->options->bfuncs()[f](cpu, 3 - f / 2, context(b, ref));
  b->to(torch::kCUDA);
  auto gpu = input.to(torch::kCUDA), gr = ref.to(torch::kCUDA);
  for (int f = 0; f < 6; ++f)
    b->options->bfuncs()[f](gpu, 3 - f / 2, context(b, gr));
  EXPECT_TRUE(torch::allclose(cpu, gpu.cpu(), 1.e-10, 1.e-10));
}

int main(int argc, char** argv) {
  torch::set_num_threads(1);
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
