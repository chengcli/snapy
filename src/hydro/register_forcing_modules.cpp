// snap
#include "hydro.hpp"

namespace snap {

std::vector<std::string> HydroImpl::register_forcings_module() {
  std::vector<std::string> forcing_names;

  if (options->grav()) {
    forcings.push_back(torch::nn::AnyModule(ConstGravity(options->grav())));
    forcing_names.push_back("const-gravity");
  }

  if (options->coriolis()) {
    forcings.push_back(torch::nn::AnyModule(Coriolis123(options->coriolis())));
    forcing_names.push_back("coriolis");
  }

  if (options->fricHeat()) {
    forcings.push_back(torch::nn::AnyModule(FricHeat(options->fricHeat())));
    forcing_names.push_back("fric-heat");
  }

  if (options->bodyHeat()) {
    forcings.push_back(torch::nn::AnyModule(BodyHeat(options->bodyHeat())));
    forcing_names.push_back("body-heat");
  }

  if (options->topCool()) {
    forcings.push_back(torch::nn::AnyModule(TopCool(options->topCool())));
    forcing_names.push_back("top-cool");
  }

  if (options->botHeat()) {
    forcings.push_back(torch::nn::AnyModule(BotHeat(options->botHeat())));
    forcing_names.push_back("bot-heat");
  }

  if (options->relaxBotComp()) {
    forcings.push_back(
        torch::nn::AnyModule(RelaxBotComp(options->relaxBotComp())));
    forcing_names.push_back("relax-bot-comp");
  }

  if (options->relaxBotTemp()) {
    forcings.push_back(
        torch::nn::AnyModule(RelaxBotTemp(options->relaxBotTemp())));
    forcing_names.push_back("relax-bot-temp");
  }

  if (options->relaxBotVelo()) {
    forcings.push_back(
        torch::nn::AnyModule(RelaxBotVelo(options->relaxBotVelo())));
    forcing_names.push_back("relax-bot-velo");
  }

  if (options->topSpongeLyr()) {
    forcings.push_back(
        torch::nn::AnyModule(TopSpongeLyr(options->topSpongeLyr())));
    forcing_names.push_back("top-sponge-lyr");
  }

  if (options->botSpongeLyr()) {
    forcings.push_back(
        torch::nn::AnyModule(BotSpongeLyr(options->botSpongeLyr())));
    forcing_names.push_back("bot-sponge-lyr");
  }

  if (options->eos()->type() == "plume-eos") {
    forcings.push_back(
        torch::nn::AnyModule(PlumeForcing(options->plumeForcing())));
    forcing_names.push_back("plume-forcing");
  }

  return forcing_names;
}
}  // namespace snap
