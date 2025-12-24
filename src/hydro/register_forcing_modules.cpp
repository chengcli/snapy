// snap
#include "hydro.hpp"

namespace snap {

std::vector<std::string> HydroImpl::_register_forcings_module() {
  std::vector<std::string> forcing_names;

  if (options->grav()) {
    forcings.push_back(torch::nn::AnyModule(ConstGravity(options->grav())));
    forcing_names.push_back("const-gravity");
  }

  if (options->coriolis()) {
    if (options->coriolis()->type() == "xyz") {
      forcings.push_back(
          torch::nn::AnyModule(CoriolisXYZ(options->coriolis(), this)));
    } else {
      forcings.push_back(
          torch::nn::AnyModule(Coriolis123(options->coriolis(), this)));
    }
    forcing_names.push_back("coriolis");
  }

  if (options->fricHeat()) {
    forcings.push_back(
        torch::nn::AnyModule(FricHeat(options->fricHeat(), this)));
    forcing_names.push_back("fric-heat");
  }

  if (options->bodyHeat()) {
    forcings.push_back(torch::nn::AnyModule(BodyHeat(options->bodyHeat())));
    forcing_names.push_back("body-heat");
  }

  if (options->topCool()) {
    forcings.push_back(torch::nn::AnyModule(TopCool(options->topCool(), this)));
    forcing_names.push_back("top-cool");
  }

  if (options->botHeat()) {
    forcings.push_back(torch::nn::AnyModule(BotHeat(options->botHeat(), this)));
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
        torch::nn::AnyModule(TopSpongeLyr(options->topSpongeLyr(), this)));
    forcing_names.push_back("top-sponge-lyr");
  }

  if (options->botSpongeLyr()) {
    forcings.push_back(
        torch::nn::AnyModule(BotSpongeLyr(options->botSpongeLyr(), this)));
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
