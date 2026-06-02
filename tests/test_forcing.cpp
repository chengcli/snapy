// external
#include <gtest/gtest.h>
#include <yaml-cpp/yaml.h>

// torch
#include <torch/torch.h>

// snap
#include <snap/snap.h>

#include <snap/forcing/forcing.hpp>
#include <snap/mesh/meshblock.hpp>

using namespace snap;

namespace {

std::shared_ptr<MeshBlockImpl> make_block() {
  return std::make_shared<MeshBlockImpl>(
      MeshBlockOptionsImpl::from_yaml("test_diffusion_moist.yaml"));
}

torch::Tensor make_primitive(std::shared_ptr<MeshBlockImpl> const& block) {
  auto coord = block->pcoord;
  auto w = torch::zeros({block->phydro->peos->nvar(), coord->options->nc3(),
                         coord->options->nc2(), coord->options->nc1()},
                        torch::kFloat64);
  w[IDN] = 1.;
  w[IVX] = 1.;
  w[IVY] = 2.;
  w[IVZ] = 3.;
  w[IPR] = 1.e5;
  w[ICY] = 0.1;
  w[ICY + 1] = 0.2;
  return w;
}

std::vector<torch::indexing::TensorIndex> bottom3(
    std::shared_ptr<MeshBlockImpl> const& block) {
  return block->part({0, 0, -1},
                     PartOptions().exterior(false).depth(1).ndim(3));
}

std::vector<torch::indexing::TensorIndex> bottom4(
    std::shared_ptr<MeshBlockImpl> const& block) {
  return block->part({0, 0, -1}, PartOptions().exterior(false).depth(1));
}

void expect_only_bottom(torch::Tensor const& du,
                        std::shared_ptr<MeshBlockImpl> const& block) {
  auto expected = torch::zeros_like(du);
  expected.index_put_(bottom4(block), du.index(bottom4(block)));
  EXPECT_TRUE(torch::allclose(du, expected));
}

}  // namespace

TEST(forcing_options, reject_invalid_values) {
  EXPECT_ANY_THROW(RelaxBotCompOptionsImpl::from_yaml(
      YAML::Load("relax-bot-comp: {tau: 0., species: [vapor], xfrac: [0.1]}")));
  EXPECT_ANY_THROW(RelaxBotCompOptionsImpl::from_yaml(
      YAML::Load("relax-bot-comp: {tau: 1., species: [vapor, vapor], xfrac: "
                 "[0.1, 0.2]}")));
  EXPECT_ANY_THROW(RelaxBotCompOptionsImpl::from_yaml(
      YAML::Load("relax-bot-comp: {tau: 1., species: [vapor], xfrac: [1.1]}")));
  EXPECT_ANY_THROW(RelaxBotTempOptionsImpl::from_yaml(
      YAML::Load("relax-bot-temp: {tau: 0., btemp: 300.}")));
  EXPECT_ANY_THROW(RelaxBotVeloOptionsImpl::from_yaml(
      YAML::Load("relax-bot-velo: {tau: 0.}")));
  EXPECT_ANY_THROW(BodyHeatOptionsImpl::from_yaml(
      YAML::Load("body-heat: {pmin: 2., pmax: 1.}")));
}

TEST(forcing_options, reject_unknown_composition_species) {
  auto block = make_block();
  auto op = RelaxBotCompOptionsImpl::from_yaml(YAML::Load(
      "relax-bot-comp: {tau: 1., species: [unknown], xfrac: [0.1]}"));
  EXPECT_ANY_THROW(RelaxBotComp(op, block->phydro.get()));
}

TEST(forcing, meshblock_registers_parent_dependent_modules) {
  auto options = MeshBlockOptionsImpl::from_yaml("test_diffusion_moist.yaml");
  options->hydro()->bodyHeat() = BodyHeatOptionsImpl::from_yaml(
      YAML::Load("body-heat: {dTdt: 1., pmin: 0., pmax: 2.e5}"));
  options->hydro()->relaxBotComp() = RelaxBotCompOptionsImpl::from_yaml(
      YAML::Load("relax-bot-comp: {tau: 1., species: [vapor], xfrac: [0.1]}"));
  options->hydro()->relaxBotTemp() = RelaxBotTempOptionsImpl::from_yaml(
      YAML::Load("relax-bot-temp: {tau: 1., btemp: 300.}"));
  options->hydro()->relaxBotVelo() = RelaxBotVeloOptionsImpl::from_yaml(
      YAML::Load("relax-bot-velo: {tau: 1.}"));

  EXPECT_NO_THROW(std::make_shared<MeshBlockImpl>(options));
}

TEST(forcing, relax_bottom_temperature) {
  auto block = make_block();
  auto w = make_primitive(block);
  auto temp = block->phydro->peos->compute("W->T", {w});
  auto du = torch::zeros_like(w);
  auto op = RelaxBotTempOptionsImpl::from_yaml(
      YAML::Load("relax-bot-temp: {tau: 2., btemp: 350.}"));

  RelaxBotTemp(op, block->phydro.get())->forward(du, w, temp, 0.5);

  auto expected = torch::zeros_like(w);
  auto bot = bottom3(block);
  auto cv = block->phydro->peos->specific_heat_cv(w, temp);
  expected[IPR].index_put_(
      bot, 0.25 * w[IDN].index(bot) * cv.index(bot) * (350. - temp.index(bot)));
  EXPECT_TRUE(torch::allclose(du, expected));
}

TEST(forcing, relax_bottom_velocity) {
  auto block = make_block();
  auto w = make_primitive(block);
  auto temp = block->phydro->peos->compute("W->T", {w});
  auto du = torch::zeros_like(w);
  auto op = RelaxBotVeloOptionsImpl::from_yaml(
      YAML::Load("relax-bot-velo: {tau: 2., bvx: 5., bvy: 6., bvz: 7.}"));

  RelaxBotVelo(op, block->phydro.get())->forward(du, w, temp, 0.5);

  auto expected = torch::zeros_like(w);
  auto bot = bottom3(block);
  expected[IVX].index_put_(bot, torch::full_like(w[IDN].index(bot), 1.));
  expected[IVY].index_put_(bot, torch::full_like(w[IDN].index(bot), 1.));
  expected[IVZ].index_put_(bot, torch::full_like(w[IDN].index(bot), 1.));
  EXPECT_TRUE(torch::allclose(du, expected));
}

TEST(forcing, relax_bottom_composition_preserves_state) {
  auto block = make_block();
  auto w = make_primitive(block);
  auto temp = block->phydro->peos->compute("W->T", {w});
  auto u = block->phydro->peos->compute("W->U", {w});
  auto du = torch::zeros_like(w);
  auto op = RelaxBotCompOptionsImpl::from_yaml(
      YAML::Load("relax-bot-comp: {tau: 2., species: [vapor], xfrac: [0.2]}"));
  RelaxBotComp forcing(op, block->phydro.get());

  forcing->forward(du, w, temp, 2.);

  auto updated = block->phydro->peos->compute("U->W", {u + du});
  auto bot = bottom4(block);
  auto bot3 = bottom3(block);
  auto updated_bot = updated.index(bot);
  auto xfrac =
      forcing->pthermo_y->compute("Y->X", {updated_bot.narrow(0, ICY, 2)});
  EXPECT_TRUE(torch::allclose(xfrac.select(-1, 1),
                              torch::full_like(xfrac.select(-1, 1), 0.2)));
  EXPECT_TRUE(torch::allclose(updated_bot[IDN], w.index(bot)[IDN]));
  EXPECT_TRUE(torch::allclose(updated_bot.narrow(0, IVX, 3),
                              w.index(bot).narrow(0, IVX, 3)));
  EXPECT_TRUE(torch::allclose(
      block->phydro->peos->compute("W->T", {updated}).index(bot3),
      temp.index(bot3), 1.e-8, 1.e-8));
  expect_only_bottom(du, block);
}

TEST(forcing, body_heat_uses_pressure_mask_and_mixture_cv) {
  auto block = make_block();
  auto w = make_primitive(block);
  auto interior = block->part({0, 0, 0}, PartOptions().exterior(false).ndim(3));
  int il = block->pcoord->il();
  w[IPR].select(-1, il + 1).fill_(2.e5);
  auto temp = block->phydro->peos->compute("W->T", {w});
  auto du = torch::zeros_like(w);
  auto op = BodyHeatOptionsImpl::from_yaml(
      YAML::Load("body-heat: {dTdt: 2., pmin: 1.5e5, pmax: 2.5e5}"));

  BodyHeat(op, block->phydro.get())->forward(du, w, temp, 0.5);

  auto expected = torch::zeros_like(w);
  auto pres = w[IPR].index(interior);
  auto cv = block->phydro->peos->specific_heat_cv(w, temp).index(interior);
  expected[IPR].index_put_(
      interior,
      torch::where(torch::logical_and(pres >= 1.5e5, pres <= 2.5e5),
                   w[IDN].index(interior) * cv, torch::zeros_like(pres)));
  EXPECT_TRUE(torch::allclose(du, expected));
}

TEST(forcing, boundary_fluxes_scale_with_timestep) {
  auto block = make_block();
  auto w = make_primitive(block);
  auto temp = block->phydro->peos->compute("W->T", {w});
  auto du1 = torch::zeros_like(w);
  auto du2 = torch::zeros_like(w);
  auto top_op = TopCoolOptionsImpl::from_yaml(
      YAML::Load("top-cool: {flux: -100., depth: 2}"));
  auto bot_op = BotHeatOptionsImpl::from_yaml(
      YAML::Load("bot-heat: {flux: 100., depth: 2}"));
  TopCool top(top_op, block->phydro.get());
  BotHeat bot(bot_op, block->phydro.get());

  top->forward(du1, w, temp, 0.25);
  bot->forward(du1, w, temp, 0.25);
  top->forward(du2, w, temp, 0.5);
  bot->forward(du2, w, temp, 0.5);

  EXPECT_TRUE(torch::allclose(du2, 2. * du1));
}
