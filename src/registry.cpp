// yaml
#include <yaml-cpp/yaml.h>

// snap
#include <snap/hydro/hydro.hpp>
#include <snap/layout/layout.hpp>
#include <snap/recon/interpolation.hpp>
#include <snap/riemann/riemann_solver.hpp>

namespace snap {

std::vector<std::string> register_forcings_module(
    HydroOptions const &opts, std::vector<torch::nn::AnyModule> &forcings) {
  std::vector<std::string> forcing_names;

  // FIXME does not work for other directions
  if (opts.grav().grav1() != 0.0 || opts.grav().grav2() != 0.0 ||
      opts.grav().grav3() != 0.0) {
    if (!opts.disable_flux_x1()) {
      forcings.push_back(torch::nn::AnyModule(ConstGravity(opts.grav())));
      forcing_names.push_back("const-gravity");
    }
  }

  if (opts.coriolis().omega1() != 0.0 || opts.coriolis().omega2() != 0.0 ||
      opts.coriolis().omega3() != 0.0) {
    forcings.push_back(torch::nn::AnyModule(Coriolis123(opts.coriolis())));
    forcing_names.push_back("coriolis");
  }

  if (opts.coriolis().omegax() != 0.0 || opts.coriolis().omegay() != 0.0 ||
      opts.coriolis().omegaz() != 0.0) {
    forcings.push_back(torch::nn::AnyModule(CoriolisXYZ(opts.coriolis())));
    if (std::find(forcing_names.begin(), forcing_names.end(), "coriolis") !=
        forcing_names.end()) {
      TORCH_CHECK(false,
                  "CoriolisXYZ cannot be used together with Coriolis123. "
                  "Please choose one of them.");
    }
    forcing_names.push_back("coriolis");
  }

  if (opts.fricHeat().grav() != 0.0) {
    forcings.push_back(torch::nn::AnyModule(FricHeat(opts.fricHeat())));
    forcing_names.push_back("fric-heat");
  }

  if (opts.bodyHeat().dTdt() != 0.0) {
    forcings.push_back(torch::nn::AnyModule(BodyHeat(opts.bodyHeat())));
    forcing_names.push_back("body-heat");
  }

  if (opts.topCool().flux() != 0.0) {
    forcings.push_back(torch::nn::AnyModule(TopCool(opts.topCool())));
    forcing_names.push_back("top-cool");
  }

  if (opts.botHeat().flux() != 0.0) {
    forcings.push_back(torch::nn::AnyModule(BotHeat(opts.botHeat())));
    forcing_names.push_back("bot-heat");
  }

  if (opts.relaxBotComp().tau() != 0.0) {
    forcings.push_back(torch::nn::AnyModule(RelaxBotComp(opts.relaxBotComp())));
    forcing_names.push_back("relax-bot-comp");
  }

  if (opts.relaxBotTemp().tau() != 0.0) {
    forcings.push_back(torch::nn::AnyModule(RelaxBotTemp(opts.relaxBotTemp())));
    forcing_names.push_back("relax-bot-temp");
  }

  if (opts.relaxBotVelo().tau() != 0.0) {
    forcings.push_back(torch::nn::AnyModule(RelaxBotVelo(opts.relaxBotVelo())));
    forcing_names.push_back("relax-bot-velo");
  }

  if (opts.topSpongeLyr().tau() != 0.0 && opts.topSpongeLyr().width() > 0.0) {
    forcings.push_back(torch::nn::AnyModule(TopSpongeLyr(opts.topSpongeLyr())));
    forcing_names.push_back("top-sponge-lyr");
  }

  if (opts.botSpongeLyr().tau() != 0.0 && opts.botSpongeLyr().width() > 0.0) {
    forcings.push_back(torch::nn::AnyModule(BotSpongeLyr(opts.botSpongeLyr())));
    forcing_names.push_back("bot-sponge-lyr");
  }

  if (opts.eos().type() == "plume-eos") {
    forcings.push_back(torch::nn::AnyModule(PlumeForcing(opts.plumeForcing())));
    forcing_names.push_back("plume-forcing");
  }

  return forcing_names;
}
}  // namespace snap
