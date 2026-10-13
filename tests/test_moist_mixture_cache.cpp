// Warm/fresh controls for the MoistMixture primitive/thermo memo.
#include <c10/core/InferenceMode.h>
#include <gtest/gtest.h>
#include <snap/snap.h>
#include <torch/torch.h>
#include <unistd.h>

#include <cstdio>
#include <fstream>
#include <snap/eos/moist_mixture.hpp>
#include <snap/hydro/hydro.hpp>
#include <snap/mesh/meshblock.hpp>
#include <string>

#include "cuda_test_gate.hpp"

using namespace snap;
namespace {
const char *config = R"(
reference-state: {Tref: 300., Pref: 1.e5}
species:
  - {name: dry, composition: {O: 0.42, N: 1.56, Ar: 0.01}, cv_R: 2.5}
  - {name: H2O, composition: {H: 2, O: 1}, cv_R: 2.5, u0_R: 0.}
  - {name: H2O(l), composition: {H: 2, O: 1}, cv_R: 9.0, u0_R: -3430.}
geometry:
  type: cartesian
  bounds: {x1min: 0., x1max: 1., x2min: 0., x2max: 1., x3min: 0., x3max: 1.}
  cells: {nx1: 7, nx2: 5, nx3: 1, nghost: 3}
dynamics:
  equation-of-state: {type: moist-mixture, density-floor: 1.e-10, pressure-floor: 1.e-10, limiter: false}
boundary-condition:
  external: {x1-inner: reflecting, x1-outer: reflecting, x2-inner: periodic, x2-outer: periodic,
             x3-inner: periodic, x3-outer: periodic}
reactions:
  - {equation: H2O <=> H2O(l), type: nucleation, rate-constant: {formula: h2o_ideal}}
)";

class MoistMixtureCache : public ::testing::TestWithParam<int> {
 protected:
  torch::Device device() const {
    return GetParam() >= 2 ? torch::kCUDA : torch::kCPU;
  }
  torch::Dtype dtype() const {
    return GetParam() % 2 ? torch::kFloat32 : torch::kFloat64;
  }
  void SetUp() override {
    if (device().is_cuda() && !snapy_cuda_test_enabled())
      GTEST_SKIP() << "snapy was not built with CUDA, or no device";
  }
  std::shared_ptr<MeshBlockImpl> block() {
    char fname[] = "./test-mixture-cache-XXXXXX";
    int fd = mkstemp(fname);
    EXPECT_NE(fd, -1);
    if (fd != -1) close(fd);
    {
      std::ofstream out(fname);
      out << config;
    }
    auto b =
        std::make_shared<MeshBlockImpl>(MeshBlockOptionsImpl::from_yaml(fname));
    std::remove(fname);
    b->to(device(), dtype());
    return b;
  }
  std::shared_ptr<MoistMixtureImpl> eos(std::shared_ptr<MeshBlockImpl> b) {
    return std::dynamic_pointer_cast<MoistMixtureImpl>(b->phydro->peos);
  }
  torch::Tensor prim(std::shared_ptr<MeshBlockImpl> b) {
    auto c = b->pcoord;
    auto w =
        torch::zeros({b->phydro->peos->nvar(), c->options->nc3(),
                      c->options->nc2(), c->options->nc1()},
                     torch::TensorOptions().device(device()).dtype(dtype()));
    auto ramp = torch::linspace(0., 1., w.size(-1), w.options());
    w[IDN] = 1.1;
    w[IPR] = 1.e5 * (1. + 0.1 * ramp);
    w[ICY] = 0.01 + 0.003 * ramp;
    w[ICY + 1] = 0.002;
    return w;
  }
  void equal_outputs(std::shared_ptr<MoistMixtureImpl> a, torch::Tensor w,
                     std::shared_ptr<MoistMixtureImpl> b, torch::Tensor fw) {
    for (auto ab : {"W->T", "W->A"}) {
      auto got = a->compute(ab, {w}).clone();
      auto want = b->compute(ab, {fw}).clone();
      ASSERT_EQ(got.sizes(), want.sizes());
      EXPECT_EQ(got.device(), want.device());
      EXPECT_EQ(got.scalar_type(), want.scalar_type());
      EXPECT_TRUE(torch::isfinite(got).all().item<bool>());
      EXPECT_TRUE(torch::equal(got, want))
          << ab
          << " max|warm-fresh|=" << (got - want).abs().max().item<double>();
    }
  }
};

TEST_P(MoistMixtureCache, one_way_dtype) {
  auto b = block(), f = block();
  auto e = eos(b), fe = eos(f);
  auto w = prim(b);
  e->compute("W->T", {w});
  auto target = dtype() == torch::kFloat64 ? torch::kFloat32 : torch::kFloat64;
  e->to(target);
  fe->to(target);
  // The caller moves its input as well as the module. Mixed-dtype inputs
  // are rejected by the thermodynamic kernels. This is a one-way control.
  equal_outputs(e, w.to(target), fe, prim(f).to(target));
}

TEST_P(MoistMixtureCache, device_move) {
  if (!snapy_cuda_test_enabled()) GTEST_SKIP() << "requires CPU and CUDA";
  auto b = block(), f = block();
  auto e = eos(b), fe = eos(f);
  auto w = prim(b);
  e->compute("W->T", {w});
  torch::Device target = device().is_cuda() ? torch::kCPU : torch::kCUDA;
  e->to(target);
  fe->to(target);
  equal_outputs(e, w.to(target), fe, prim(f).to(target));
}

TEST_P(MoistMixtureCache, inference_input_edit) {
  auto b = block(), f = block();
  auto e = eos(b), fe = eos(f);
  c10::InferenceMode guard;
  auto w = prim(b), fw = prim(f);
  ASSERT_TRUE(w.is_inference());
  auto before = e->compute("W->T", {w}).clone();
  w[IPR].mul_(1.01);
  fw[IPR].mul_(1.01);
  auto after = e->compute("W->T", {w}).clone();
  EXPECT_FALSE(torch::equal(before, after));
  equal_outputs(e, w, fe, fw);
}

TEST_P(MoistMixtureCache, in_place_inv_mu) {
  auto b = block(), f = block();
  auto e = eos(b), fe = eos(f);
  auto w = prim(b);
  auto before = e->compute("W->T", {w}).clone();
  e->pthermo->inv_mu.mul_(1.01);
  fe->pthermo->inv_mu.mul_(1.01);
  EXPECT_FALSE(torch::equal(before, e->compute("W->T", {w})));
  equal_outputs(e, w, fe, prim(f));
}

TEST_P(MoistMixtureCache, externally_mutable_options) {
  auto b = block(), f = block();
  auto e = eos(b), fe = eos(f);
  auto w = prim(b);
  // Retain an externally mutable vector reference across the cache fill.
  auto &vapor_ids = e->pthermo->options->vapor_ids();
  auto before = e->compute("W->T", {w}).clone();
  // Reclassify H2O as condensed in both thermodynamic tables, preserving
  // species order, total species count, and the primitive inputs.
  auto id = vapor_ids.back();
  vapor_ids.pop_back();
  e->pthermo->options->cloud_ids().insert(
      e->pthermo->options->cloud_ids().begin(), id);
  auto &fresh_vapor_ids = fe->pthermo->options->vapor_ids();
  auto fresh_id = fresh_vapor_ids.back();
  fresh_vapor_ids.pop_back();
  fe->pthermo->options->cloud_ids().insert(
      fe->pthermo->options->cloud_ids().begin(), fresh_id);
  EXPECT_FALSE(torch::equal(before, e->compute("W->T", {w})));
  equal_outputs(e, w, fe, prim(f));
}

TEST_P(MoistMixtureCache, matching_unsupported_derivative) {
  auto b = block(), f = block();
  auto e = eos(b), fe = eos(f);
  auto w = prim(b), fw = prim(f);
  e->compute("W->T", {w});
  w.requires_grad_(true);
  fw.requires_grad_(true);
  auto attempt = [](std::shared_ptr<MoistMixtureImpl> m, torch::Tensor x) {
    try {
      m->compute("W->T", {x}).sum().backward();
    } catch (c10::Error const &err) {
      return std::string(err.what_without_backtrace());
    }
    return std::string("NO ERROR");
  };
  auto warm_error = attempt(e, w);
  auto fresh_error = attempt(fe, fw);
  EXPECT_NE(fresh_error.find("derivative for set_ is not implemented"),
            std::string::npos)
      << fresh_error;
  EXPECT_EQ(warm_error, fresh_error);
  // Repeating on the same input must not turn into a consumed-graph error.
  EXPECT_EQ(attempt(e, w), fresh_error);
}

// Repeated primitive misses must not hide a later dependency change. The
// control always receives a distinct equal primitive, forcing fresh evaluation.
TEST_P(MoistMixtureCache, primitive_refresh_then_dependency_transitions) {
  auto b = block(), f = block();
  auto e = eos(b), fe = eos(f);
  auto w = prim(b);
  auto &vapor_ids = e->pthermo->options->vapor_ids();
  e->compute("W->T", {w});
  for (int i = 0; i < 3; ++i) {
    w[IPR].mul_(1.001);
    equal_outputs(e, w, fe, w.clone());
  }

  // This reference predates all the primitive-version misses above.
  auto id = vapor_ids.back();
  vapor_ids.pop_back();
  e->pthermo->options->cloud_ids().insert(
      e->pthermo->options->cloud_ids().begin(), id);
  auto &fresh_ids = fe->pthermo->options->vapor_ids();
  auto fresh_id = fresh_ids.back();
  fresh_ids.pop_back();
  fe->pthermo->options->cloud_ids().insert(
      fe->pthermo->options->cloud_ids().begin(), fresh_id);
  equal_outputs(e, w, fe, w.clone());

  w[IPR].mul_(1.001);
  equal_outputs(e, w, fe, w.clone());
  e->pthermo->inv_mu.mul_(1.01);
  fe->pthermo->inv_mu.mul_(1.01);
  equal_outputs(e, w, fe, w.clone());

  auto other = dtype() == torch::kFloat64 ? torch::kFloat32 : torch::kFloat64;
  e->pthermo->to(other);
  e->pthermo->to(dtype());
  fe->pthermo->to(other);
  fe->pthermo->to(dtype());
  equal_outputs(e, w, fe, w.clone());
  e->to(other);
  e->to(dtype());
  fe->to(other);
  fe->to(dtype());
  equal_outputs(e, w, fe, w.clone());

  {
    c10::InferenceMode guard;
    auto iw = w.clone();
    ASSERT_TRUE(iw.is_inference());
    e->compute("W->T", {iw});
    iw[IPR].mul_(1.001);
    equal_outputs(e, iw, fe, iw.clone());
  }
  w[IPR].mul_(1.001);
  equal_outputs(e, w, fe, w.clone());

  // The same warmed module now sees a grad-enabled key. Derivatives remain
  // unsupported; compare errors to fresh evaluation and repeat the attempt.
  w.requires_grad_(true);
  auto fw = w.detach().clone().requires_grad_(true);
  auto attempt = [](std::shared_ptr<MoistMixtureImpl> m, torch::Tensor x) {
    try {
      m->compute("W->T", {x}).sum().backward();
    } catch (c10::Error const &err) {
      return std::string(err.what_without_backtrace());
    }
    return std::string("NO ERROR");
  };
  auto expected = attempt(fe, fw);
  ASSERT_NE(expected.find("derivative for set_ is not implemented"),
            std::string::npos)
      << expected;
  EXPECT_EQ(attempt(e, w), expected);
  EXPECT_EQ(attempt(e, w), expected);
  w.requires_grad_(false);
  {
    torch::NoGradGuard guard;
    w[IPR].mul_(1.001);
    equal_outputs(e, w, fe, w.clone());
  }
}

// With the limiter disabled U->W marks its freshly computed values directly,
// so it also reaches _mark_cache without going through _ensure_cache.
TEST_P(MoistMixtureCache, conserved_refresh_then_dependency_edits) {
  auto b = block(), f = block();
  auto e = eos(b), fe = eos(f);
  ASSERT_FALSE(e->options->limiter());
  auto u = e->compute("W->U", {prim(b)}).clone();
  auto &coefficients = e->pthermo->options->cref_R();
  torch::Tensor w;
  for (int i = 0; i < 3; ++i) {
    u[IPR].mul_(1.001);
    w = e->compute("U->W", {u.clone()});
    f = block();
    fe = eos(f);
    auto fw = fe->compute("U->W", {u.clone()});
    EXPECT_TRUE(torch::equal(w, fw));
    equal_outputs(e, w, fe, fw);
  }
  coefficients[0] *= 1.02;
  fe->pthermo->options->cref_R()[0] *= 1.02;
  e->pthermo->inv_mu.mul_(1.01);
  fe->pthermo->inv_mu.mul_(1.01);
  equal_outputs(e, w, fe, w.clone());
  // Exercise direct cache marking again after the dependency edits.
  w = e->compute("U->W", {u.clone()});
  auto fw = fe->compute("U->W", {u.clone()});
  EXPECT_TRUE(torch::equal(w, fw));
  equal_outputs(e, w, fe, fw);
}

INSTANTIATE_TEST_SUITE_P(CpuCudaFloat64Float32, MoistMixtureCache,
                         ::testing::Values(0, 1, 2, 3));
}  // namespace
