// Cache lifecycle across Module::to (#297/#308/#309 grid caches).
//
// torch::nn::Module::to converts parameters and buffers with set_data
// (torch/csrc/api/include/torch/nn/module.h), which KEEPS the TensorImpl
// identity and does NOT bump the version counter while replacing the values.
// Module::to_impl also recurses into children through their VIRTUAL to(), so a
// parent conversion reaches every child's caches.
//
// A grid cache keyed on (tensor identity, version, dtype, device) therefore
// rebuilds on a ONE-WAY conversion -- the dtype term catches it -- but is blind
// to a ROUND TRIP: after float64 -> float32 -> float64 the dtype is float64
// again, identity and version are unchanged, and the cache returns a value
// built from the ORIGINAL full-precision grid while the grid itself now holds
// float32-rounded numbers.
//
// Each test is a control/variable pair on a production entry point. The control
// is a second block that reaches the same final grid WITHOUT a populated cache,
// so control and variable must agree exactly.
//
// Grid spacings here are deliberately NOT representable in float32 (1/3 of a
// unit box), so a stale cache is observable rather than accidentally equal.

// external
#include <gtest/gtest.h>

// C/C++
#include <cstdio>
#include <fstream>
#include <string>

// POSIX
#include <unistd.h>

// torch
#include <c10/core/InferenceMode.h>
#include <torch/torch.h>

// snap
#include <snap/snap.h>

#include <snap/coord/coordinate.hpp>
#include <snap/eos/equation_of_state.hpp>
#include <snap/eos/ideal_moist.hpp>
#include <snap/hydro/hydro.hpp>
#include <snap/mesh/meshblock.hpp>

// tests
#include "cuda_test_gate.hpp"

using namespace snap;

namespace {

// dx = 1/3 in every direction: not representable in float32, so a float32
// excursion is visible in the geometry
const char* cartesian_config = R"(
reference-state:
  Tref: 300.
  Pref: 1.e5

species:
  - name: dry
    composition: {O: 0.42, N: 1.56, Ar: 0.01}
    cv_R: 2.5
  - name: vapor
    composition: {H: 2, O: 1}
    cv_R: 3.5

  - name: cloud
    composition: {H: 2, O: 1}
    cv_R: 9.0
    u0_R: -3430.

reactions:
  - equation: vapor <=> cloud
    type: nucleation
    rate-constant: {formula: h2o_ideal}

dynamics:
  equation-of-state:
    type: ideal-moist

forcing:
  const-gravity:
    grav1: -10.

geometry:
  type: cartesian
  cells: {nx1: 3, nx2: 3, nx3: 3, nghost: 1}
  bounds: {x1min: 0., x1max: 1., x2min: 0., x2max: 1., x3min: 0., x3max: 1.}

boundary-condition:
  external:
    x1-inner: reflecting
    x1-outer: reflecting
)";

std::string write_temp_config(char const* config) {
  char fname[] = "/tmp/test-cache-lifecycle-XXXXXX";
  int fd = mkstemp(fname);
  EXPECT_NE(fd, -1);
  if (fd != -1) close(fd);
  std::ofstream outfile(fname);
  outfile << config;
  outfile.close();
  return fname;
}

std::shared_ptr<MeshBlockImpl> make_block() {
  auto fname = write_temp_config(cartesian_config);
  auto block =
      std::make_shared<MeshBlockImpl>(MeshBlockOptionsImpl::from_yaml(fname));
  std::remove(fname.c_str());
  return block;
}

// a resting moist state, identical for every block
Variables initial_state(std::shared_ptr<MeshBlockImpl> const& block,
                        torch::Dtype dtype,
                        torch::Device device = torch::kCPU) {
  auto coord = block->pcoord;
  auto w = torch::zeros({block->phydro->peos->nvar(), coord->options->nc3(),
                         coord->options->nc2(), coord->options->nc1()},
                        torch::TensorOptions().dtype(dtype).device(device));
  w[IDN].fill_(1.);
  w[IPR].fill_(1.e5);
  w[ICY].fill_(0.01);
  w[ICY + 1].fill_(0.02);
  // a vertical gradient, so the x1 stencils and the gravity work are exercised
  for (int i = 0; i < coord->options->nc1(); ++i)
    w[IPR].select(-1, i) *= 1. + 0.1 * i;
  Variables vars;
  vars["hydro_w"] = w;
  block->initialize(vars);
  return vars;
}

// touch every cached geometry accessor a solver uses
std::vector<torch::Tensor> geometry(Coordinate const& c) {
  return {c->face_area1(), c->face_area2(), c->face_area3(), c->cell_volume()};
}

void expect_all_equal(std::vector<torch::Tensor> const& got,
                      std::vector<torch::Tensor> const& want,
                      char const* what) {
  ASSERT_EQ(got.size(), want.size());
  for (size_t i = 0; i < got.size(); ++i) {
    EXPECT_EQ(got[i].scalar_type(), want[i].scalar_type())
        << what << " [" << i << "]";
    EXPECT_EQ(got[i].device(), want[i].device()) << what << " [" << i << "]";
    double diff = (got[i].to(torch::kFloat64) - want[i].to(torch::kFloat64))
                      .abs()
                      .max()
                      .item<double>();
    EXPECT_TRUE(torch::equal(got[i], want[i]))
        << what << " [" << i << "] max|diff| = " << diff;
  }
}

}  // namespace

// ---------------------------------------------------------------------------
// CoordinateImpl::GeomCache (#309), reached through a parent MeshBlock::to
// ---------------------------------------------------------------------------

// THE REGRESSION. Populate, round trip, read again. The control block reaches
// the same float32-rounded grid with no cache populated beforehand.
TEST(CacheLifecycleTo, geometry_survives_f64_f32_f64_round_trip) {
  auto variable = make_block();
  geometry(variable->pcoord);  // populate at the original float64 grid
  variable->to(torch::kFloat32);
  variable->to(torch::kFloat64);

  auto control = make_block();
  control->to(torch::kFloat32);
  control->to(torch::kFloat64);  // same final grid, cache never populated

  // the grids themselves must agree -- otherwise the test is not comparing
  // like with like
  ASSERT_TRUE(torch::equal(variable->pcoord->dx1f, control->pcoord->dx1f));
  ASSERT_TRUE(torch::equal(variable->pcoord->dx2f, control->pcoord->dx2f));

  expect_all_equal(geometry(variable->pcoord), geometry(control->pcoord),
                   "round trip");
}

// A conversion to the dtype the module already has must not corrupt anything.
TEST(CacheLifecycleTo, geometry_survives_noop_conversion) {
  auto variable = make_block();
  auto before = geometry(variable->pcoord);
  variable->to(torch::kFloat64);  // already float64: a no-op
  expect_all_equal(geometry(variable->pcoord), before, "no-op conversion");
}

// The one-way case the existing dtype guard already covers: keep it covered.
TEST(CacheLifecycleTo, geometry_follows_one_way_conversion) {
  auto variable = make_block();
  geometry(variable->pcoord);
  variable->to(torch::kFloat32);

  auto control = make_block();
  control->to(torch::kFloat32);

  auto got = geometry(variable->pcoord);
  for (auto const& g : got) ASSERT_EQ(g.scalar_type(), torch::kFloat32);
  expect_all_equal(got, geometry(control->pcoord), "one way");
}

// Converting one block must not disturb another.
TEST(CacheLifecycleTo, geometry_two_instances_are_independent) {
  auto a = make_block();
  auto b = make_block();
  auto b_before = geometry(b->pcoord);

  geometry(a->pcoord);
  a->to(torch::kFloat32);
  a->to(torch::kFloat64);

  expect_all_equal(geometry(b->pcoord), b_before, "other instance");
}

// Module::to_impl recurses into children through their virtual to(): a
// conversion requested on the PARENT must reach the coordinate child's cache.
// (The tests above already go through MeshBlock; this pins the child-only
// conversion as well, so the hook cannot be installed on the parent alone.)
TEST(CacheLifecycleTo, geometry_round_trip_applied_to_the_child_module) {
  auto variable = make_block();
  auto coord = std::dynamic_pointer_cast<torch::nn::Module>(
      std::dynamic_pointer_cast<CartesianImpl>(variable->pcoord));
  ASSERT_NE(coord, nullptr) << "cartesian coordinate expected";

  geometry(variable->pcoord);
  coord->to(torch::kFloat32);
  coord->to(torch::kFloat64);

  auto control = make_block();
  auto control_coord = std::dynamic_pointer_cast<torch::nn::Module>(
      std::dynamic_pointer_cast<CartesianImpl>(control->pcoord));
  ASSERT_NE(control_coord, nullptr);
  control_coord->to(torch::kFloat32);
  control_coord->to(torch::kFloat64);

  expect_all_equal(geometry(variable->pcoord), geometry(control->pcoord),
                   "child round trip");
}

// ---------------------------------------------------------------------------
// The hydro, implicit and EOS caches, through a real step
// (HydroImpl::_s2/_cell_vol/_pe_work/wb_ref4_/x1pm_/x1src_,
//  ImplicitHydroImpl::_g_*/_rx_*, IdealMoistImpl::_cc)
// ---------------------------------------------------------------------------

// Step once to populate every per-step cache, round trip, then step again from
// an identical freshly initialised state. The control block round-trips first
// and takes the same step with no cache carried across the conversion.
TEST(CacheLifecycleTo, step_after_round_trip_matches_control) {
  auto variable = make_block();
  {
    auto warm = initial_state(variable, torch::kFloat64);
    variable->phydro->forward(1.e-3, warm.at("hydro_u"), warm);
  }
  variable->to(torch::kFloat32);
  variable->to(torch::kFloat64);
  auto vvars = initial_state(variable, torch::kFloat64);
  variable->phydro->forward(1.e-3, vvars.at("hydro_u"), vvars);

  auto control = make_block();
  control->to(torch::kFloat32);
  control->to(torch::kFloat64);
  auto cvars = initial_state(control, torch::kFloat64);
  control->phydro->forward(1.e-3, cvars.at("hydro_u"), cvars);

  auto const& got = vvars.at("hydro_u");
  auto const& want = cvars.at("hydro_u");
  double diff = (got - want).abs().max().item<double>();
  EXPECT_TRUE(torch::equal(got, want))
      << "step after round trip: max|diff| = " << diff;
}

// IdealMoistImpl::_cc is keyed on the species-table BUFFERS, which Module::to
// does convert, so it is exposed in the same way.
TEST(CacheLifecycleTo, species_enthalpy_after_round_trip_matches_control) {
  auto variable = make_block();
  auto vvars = initial_state(variable, torch::kFloat64);
  variable->phydro->peos->species_enthalpy(vvars.at("hydro_w"));
  variable->to(torch::kFloat32);
  variable->to(torch::kFloat64);
  auto vvars2 = initial_state(variable, torch::kFloat64);
  auto got = variable->phydro->peos->species_enthalpy(vvars2.at("hydro_w"));

  auto control = make_block();
  control->to(torch::kFloat32);
  control->to(torch::kFloat64);
  auto cvars = initial_state(control, torch::kFloat64);
  auto want = control->phydro->peos->species_enthalpy(cvars.at("hydro_w"));

  ASSERT_EQ(got.scalar_type(), want.scalar_type());
  double diff = (got - want).abs().max().item<double>();
  EXPECT_TRUE(torch::equal(got, want))
      << "species_enthalpy after round trip: max|diff| = " << diff;
}

// IdealMoistImpl::_temp_memo: its KEY is the caller's primitive state, which
// Module::to never converts, so the dtype/device mechanism that applies to the
// other caches does not apply to the key. Its VALUE, however, is
// prim[IPR] / (prim[IDN] * Rd * f_eps(yfrac)), and f_eps multiplies by
// inv_mu_ratio_m1 -- a registered BUFFER that Module::to does convert. So the
// memo still has to be dropped when the EOS is converted, for a reason that
// has nothing to do with its key.
//
// Both halves are pinned here: the key really is not a module buffer, and the
// memoised value must follow a conversion of the species table.
TEST(CacheLifecycleTo, temperature_memo_key_is_not_a_module_buffer) {
  auto block = make_block();
  auto eos = block->phydro->peos;
  auto vars = initial_state(block, torch::kFloat64);
  auto w = vars.at("hydro_w");

  auto moist = std::dynamic_pointer_cast<IdealMoistImpl>(eos);
  ASSERT_NE(moist, nullptr) << "ideal-moist EOS expected";
  for (auto const& named : moist->named_buffers(/*recurse=*/true))
    EXPECT_FALSE(named.value().is_same(w))
        << "the primitive state is a registered buffer named " << named.key()
        << "; the memo's KEY would then be converted too";
}

TEST(CacheLifecycleTo, temperature_memo_follows_a_converted_species_table) {
  auto variable = make_block();
  auto vvars = initial_state(variable, torch::kFloat64);
  auto w = vvars.at("hydro_w");
  variable->phydro->peos->compute("W->T", {w});  // memoise against this w
  variable->to(torch::kFloat32);
  variable->to(torch::kFloat64);
  // the SAME caller tensor: the key still matches, but inv_mu_ratio_m1 has
  // been through float32 and the memoised temperature was built before that
  auto got = variable->phydro->peos->compute("W->T", {w});

  auto control = make_block();
  control->to(torch::kFloat32);
  control->to(torch::kFloat64);
  auto cvars = initial_state(control, torch::kFloat64);
  auto want = control->phydro->peos->compute("W->T", {cvars.at("hydro_w")});

  ASSERT_TRUE(torch::equal(w, cvars.at("hydro_w")));
  double diff = (got - want).abs().max().item<double>();
  EXPECT_TRUE(torch::equal(got, want))
      << "memoised temperature after a species-table round trip: max|diff| = "
      << diff;
}

// ---------------------------------------------------------------------------
// Inference tensors as cache keys (#308/#309)
// ---------------------------------------------------------------------------
//
// A tensor made under InferenceMode keeps NO version counter. Reading
// `_version()` on one throws, and mapping it to a constant instead lets an
// in-place write through unnoticed -- a silent stale value, which is worse
// than the throw. Such a key is simply uncacheable: the value has to be
// rebuilt on every call.

// Mapping an inference tensor's version to a constant hides a real edit.
TEST(CacheLifecycleInference,
     cache_follows_an_in_place_edit_of_an_inference_grid) {
  c10::InferenceMode guard;
  auto block = make_block();
  auto coord = block->pcoord;
  ASSERT_TRUE(coord->dx1f.is_inference());

  auto before = coord->cell_volume().clone();
  coord->dx1f.mul_(2.0);  // the grid really moved
  auto after = coord->cell_volume();

  EXPECT_FALSE(torch::equal(before, after))
      << "the cell volume did not follow an in-place edit of dx1f";
}

// The control: the same edit on an ordinary (non-inference) grid is followed,
// so the test above is about inference tensors and not about the cache in
// general.
TEST(CacheLifecycleInference,
     cache_follows_an_in_place_edit_of_an_ordinary_grid) {
  auto block = make_block();
  auto coord = block->pcoord;
  ASSERT_FALSE(coord->dx1f.is_inference());

  auto before = coord->cell_volume().clone();
  coord->dx1f.mul_(2.0);
  auto after = coord->cell_volume();

  EXPECT_FALSE(torch::equal(before, after));
}

// A step with a block built under InferenceMode reaches the raw _version()
// calls of #309 (hydro_forward, implicit_hydro, ideal_moist), which throw.
// Inference mode is a supported way to run a model, so the step must work.
TEST(CacheLifecycleInference, a_step_runs_with_an_inference_built_block) {
  c10::InferenceMode guard;
  auto block = make_block();
  ASSERT_TRUE(block->pcoord->x1f.is_inference());
  auto vars = initial_state(block, torch::kFloat64);
  EXPECT_NO_THROW(block->phydro->forward(1.e-3, vars.at("hydro_u"), vars));
}

// ---------------------------------------------------------------------------
// The other direction on CPU, and the same thing on real CUDA
// ---------------------------------------------------------------------------

// Starting from float32 and coming back: the dtype term of every key returns
// to float32 just as it returns to float64 the other way round.
TEST(CacheLifecycleTo, geometry_survives_f32_f64_f32_round_trip) {
  auto variable = make_block();
  variable->to(torch::kFloat32);
  geometry(variable->pcoord);  // populate at float32
  variable->to(torch::kFloat64);
  variable->to(torch::kFloat32);

  auto control = make_block();
  control->to(torch::kFloat32);
  control->to(torch::kFloat64);
  control->to(torch::kFloat32);

  expect_all_equal(geometry(variable->pcoord), geometry(control->pcoord),
                   "f32 round trip");
}

// A CPU build of snapy must not enter a CUDA test just because a GPU is
// visible: snapy_cuda_test_enabled() checks the BUILD, not only the device.
#define SKIP_WITHOUT_CUDA()       \
  if (!snapy_cuda_test_enabled()) \
  GTEST_SKIP() << "snapy was not built with CUDA, or no device"

// A DEVICE round trip, CPU -> CUDA -> CPU. Module::to keeps the buffers'
// identity and version here too, and the device term of the key comes back to
// where it started.
TEST(CacheLifecycleToCuda, geometry_survives_cpu_cuda_cpu_round_trip) {
  SKIP_WITHOUT_CUDA();
  auto variable = make_block();
  geometry(variable->pcoord);
  variable->to(torch::kCUDA);
  variable->to(torch::kCPU);

  auto control = make_block();
  control->to(torch::kCUDA);
  control->to(torch::kCPU);

  expect_all_equal(geometry(variable->pcoord), geometry(control->pcoord),
                   "device round trip");
}

// A dtype round trip performed while the block lives on the device.
TEST(CacheLifecycleToCuda, geometry_survives_dtype_round_trip_on_device) {
  SKIP_WITHOUT_CUDA();
  auto variable = make_block();
  variable->to(torch::kCUDA);
  geometry(variable->pcoord);
  variable->to(torch::kFloat32);
  variable->to(torch::kFloat64);

  auto control = make_block();
  control->to(torch::kCUDA);
  control->to(torch::kFloat32);
  control->to(torch::kFloat64);

  auto got = geometry(variable->pcoord);
  for (auto const& g : got) ASSERT_TRUE(g.is_cuda());
  expect_all_equal(got, geometry(control->pcoord), "device dtype round trip");
}

// The EOS species table, on the device.
TEST(CacheLifecycleToCuda, species_enthalpy_after_round_trip_on_device) {
  SKIP_WITHOUT_CUDA();
  auto variable = make_block();
  variable->to(torch::kCUDA);
  auto vvars = initial_state(variable, torch::kFloat64, torch::kCUDA);
  variable->phydro->peos->species_enthalpy(vvars.at("hydro_w"));
  variable->to(torch::kFloat32);
  variable->to(torch::kFloat64);
  auto vvars2 = initial_state(variable, torch::kFloat64, torch::kCUDA);
  auto got = variable->phydro->peos->species_enthalpy(vvars2.at("hydro_w"));

  auto control = make_block();
  control->to(torch::kCUDA);
  control->to(torch::kFloat32);
  control->to(torch::kFloat64);
  auto cvars = initial_state(control, torch::kFloat64, torch::kCUDA);
  auto want = control->phydro->peos->species_enthalpy(cvars.at("hydro_w"));

  ASSERT_TRUE(got.is_cuda());
  double diff = (got - want).abs().max().item<double>();
  EXPECT_TRUE(torch::equal(got, want))
      << "species_enthalpy on device after round trip: max|diff| = " << diff;
}

// A real step on the device after a round trip.
TEST(CacheLifecycleToCuda, step_after_round_trip_matches_control_on_device) {
  SKIP_WITHOUT_CUDA();
  auto variable = make_block();
  variable->to(torch::kCUDA);
  {
    auto warm = initial_state(variable, torch::kFloat64, torch::kCUDA);
    variable->phydro->forward(1.e-3, warm.at("hydro_u"), warm);
  }
  variable->to(torch::kFloat32);
  variable->to(torch::kFloat64);
  auto vvars = initial_state(variable, torch::kFloat64, torch::kCUDA);
  variable->phydro->forward(1.e-3, vvars.at("hydro_u"), vvars);

  auto control = make_block();
  control->to(torch::kCUDA);
  control->to(torch::kFloat32);
  control->to(torch::kFloat64);
  auto cvars = initial_state(control, torch::kFloat64, torch::kCUDA);
  control->phydro->forward(1.e-3, cvars.at("hydro_u"), cvars);

  auto const& got = vvars.at("hydro_u");
  auto const& want = cvars.at("hydro_u");
  double diff = (got - want).abs().max().item<double>();
  EXPECT_TRUE(torch::equal(got, want))
      << "device step after round trip: max|diff| = " << diff;
}
