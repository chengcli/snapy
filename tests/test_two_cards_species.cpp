// C/C++
#include <sstream>

// external
#include <gtest/gtest.h>

// torch
#include <torch/torch.h>

// kintera
#include <kintera/species.hpp>

// snap
#include <snap/snap.h>

#include <snap/mesh/meshblock.hpp>
#include <snap/sedimentation/sedimentation.hpp>

// tests
#include "device_testing.hpp"

using namespace snap;

// kintera's species table (species_names, species_weights) is process-global
// and refills when a card with a different species list is loaded. A block
// built from card A must keep card A's species whatever card is loaded later:
// the conserved limiter borrows a negative cloud from A's parent vapors with
// A's mass fractions, and sedimentation names A's particles.
//
// Card A: vapors H2O, NH3, H2S; clouds H2O(l) (parent H2O) and NH4SH (parents
// NH3 + H2S). Card B holds the same species in another order.
namespace {

char const* kCardA = "test_two_cards_species_a.yaml";
char const* kCardB = "test_two_cards_species_b.yaml";
int const kVapor = 3, kCloud = 2;
double const kDeficit = 0.05;

// loads card B and checks that it really refilled the global table
void load_card_b() {
  MeshBlockOptionsImpl::from_yaml(kCardB);
  ASSERT_EQ(kintera::species_names.size(), 6u);
  ASSERT_EQ(kintera::species_names[1], "NH3")
      << "card B did not refill kintera's species table (needs kintera #121)";
}

std::string str(std::vector<double> const& v) {
  std::ostringstream os;
  for (auto x : v) os << x << " ";
  return os.str();
}

// debit of each vapor slot (H2O, NH3, H2S) per unit deficit of cloud j, i.e.
// the parent slots and mass fractions the limiter used for that cloud
std::vector<double> debit(MeshBlockImpl& block, int j, torch::Device device,
                          torch::Dtype dtype) {
  auto coord = block.pcoord;
  auto cons = torch::zeros({block.phydro->peos->nvar(), coord->options->nc3(),
                            coord->options->nc2(), coord->options->nc1()},
                           torch::device(device).dtype(dtype));
  double vapor[kVapor] = {0.4, 0.3, 0.2};
  cons[IDN].fill_(1.);
  cons[IPR].fill_(1.e8);
  for (int i = 0; i < kVapor; ++i) cons[ICY + i].fill_(vapor[i]);
  cons[ICY + kVapor + j].fill_(-kDeficit);  // over-drained by the tracer flux

  block.phydro->peos->apply_conserved_limiter_(cons);

  std::vector<double> out;
  for (int i = 0; i < kVapor; ++i) {
    out.push_back((vapor[i] - cons[ICY + i].mean().item<double>()) / kDeficit);
  }
  return out;
}

// A's parent map: H2O(l) from H2O alone; NH4SH from NH3 and H2S by molar mass
void expect_card_a_parents(MeshBlockImpl& block, torch::Device device,
                           torch::Dtype dtype) {
  auto peos = block.phydro->peos;
  double w_nh3 = peos->species_weight(2), w_h2s = peos->species_weight(3);
  std::vector<std::vector<double>> expected = {
      {1., 0., 0.}, {0., w_nh3 / (w_nh3 + w_h2s), w_h2s / (w_nh3 + w_h2s)}};
  char const* cloud[kCloud] = {"H2O(l)", "NH4SH"};
  double tol = dtype == torch::kFloat64 ? 1.e-10 : 1.e-4;

  for (int j = 0; j < kCloud; ++j) {
    auto got = debit(block, j, device, dtype);
    for (int i = 0; i < kVapor; ++i) {
      EXPECT_NEAR(got[i], expected[j][i], tol)
          << "cloud " << cloud[j] << ": debit per unit deficit of (H2O NH3 "
          << "H2S) = " << str(got) << "instead of " << str(expected[j]);
    }
  }
}

std::shared_ptr<MeshBlockImpl> make_block(MeshBlockOptions options,
                                          torch::Device device,
                                          torch::Dtype dtype) {
  auto block = std::make_shared<MeshBlockImpl>(options);
  block->to(device, dtype);
  return block;
}

// (a) block A is built, then card B is loaded, then block A is used
void limiter_after_block_built(bool second_card, torch::Device device,
                               torch::Dtype dtype) {
  auto block =
      make_block(MeshBlockOptionsImpl::from_yaml(kCardA), device, dtype);
  if (second_card) load_card_b();
  expect_card_a_parents(*block, device, dtype);
}

void sedvel_after_block_built(bool second_card, torch::Device device,
                              torch::Dtype dtype) {
  auto block =
      make_block(MeshBlockOptionsImpl::from_yaml(kCardA), device, dtype);
  if (second_card) load_card_b();
  auto names = block->phydro->psed->options->sedvel()->species();
  std::vector<std::string> expected = {"H2O(l)", "NH4SH"};
  std::string got;
  for (auto const& n : names) got += n + " ";
  EXPECT_EQ(names, expected) << "sedimentation species() = [ " << got
                             << "] instead of [ H2O(l) NH4SH ]";
}

// (b) card A's options are parsed, then card B is loaded, then block A is
// built from A's options
void limiter_block_built_after_second_card(bool second_card,
                                           torch::Device device,
                                           torch::Dtype dtype) {
  auto options = MeshBlockOptionsImpl::from_yaml(kCardA);
  if (second_card) load_card_b();
  auto block = make_block(options, device, dtype);
  expect_card_a_parents(*block, device, dtype);
}

}  // namespace

TEST_P(DeviceTest, two_cards_limiter_after_block_built) {
  limiter_after_block_built(true, device, dtype);
}

TEST_P(DeviceTest, two_cards_limiter_after_block_built_control) {
  limiter_after_block_built(false, device, dtype);
}

TEST_P(DeviceTest, two_cards_sedvel_after_block_built) {
  sedvel_after_block_built(true, device, dtype);
}

TEST_P(DeviceTest, two_cards_sedvel_after_block_built_control) {
  sedvel_after_block_built(false, device, dtype);
}

TEST_P(DeviceTest, two_cards_limiter_block_built_after_second_card) {
  limiter_block_built_after_second_card(true, device, dtype);
}

TEST_P(DeviceTest, two_cards_limiter_block_built_after_second_card_control) {
  limiter_block_built_after_second_card(false, device, dtype);
}
