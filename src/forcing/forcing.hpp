#pragma once

// torch
#include <torch/nn/cloneable.h>
#include <torch/nn/module.h>
#include <torch/nn/modules/common.h>
#include <torch/nn/modules/container/any.h>

// kintera
#include <kintera/utils/format.hpp>

// snap
#include <snap/coord/coordinate.hpp>
#include <snap/sedimentation/sedimentation.hpp>

// arg
#include <snap/add_arg.h>

namespace YAML {
class Node;
}  // namespace YAML

namespace snap {

struct ConstGravityOptions {
  static ConstGravityOptions from_yaml(YAML::Node const& node);
  ConstGravityOptions() = default;
  void report(std::ostream& os) const {
    os << "* grav1 = " << grav1() << "\n"
       << "* grav2 = " << grav2() << "\n"
       << "* grav3 = " << grav3() << "\n";
  }

  ADD_ARG(double, grav1) = 0.;
  ADD_ARG(double, grav2) = 0.;
  ADD_ARG(double, grav3) = 0.;
};

struct CoriolisOptions {
  static CoriolisOptions from_yaml(YAML::Node const& node);
  CoriolisOptions() = default;
  void report(std::ostream& os) const {
    os << "* omega1 = " << omega1() << "\n"
       << "* omega2 = " << omega2() << "\n"
       << "* omega3 = " << omega3() << "\n"
       << "* omegax = " << omegax() << "\n"
       << "* omegay = " << omegay() << "\n"
       << "* omegaz = " << omegaz() << "\n";
  }

  ADD_ARG(double, omega1) = 0.;
  ADD_ARG(double, omega2) = 0.;
  ADD_ARG(double, omega3) = 0.;

  ADD_ARG(double, omegax) = 0.;
  ADD_ARG(double, omegay) = 0.;
  ADD_ARG(double, omegaz) = 0.;

  ADD_ARG(CoordinateOptions, coord);
};

struct DiffusionOptions {
  static DiffusionOptions from_yaml(YAML::Node const& node);
  DiffusionOptions() = default;
  void report(std::ostream& os) const {
    os << "* K = " << K() << "\n"
       << "* type = " << type() << "\n";
  }

  ADD_ARG(double, K) = 0.;
  ADD_ARG(std::string, type) = "theta";
};

struct FricHeatOptions {
  static FricHeatOptions from_yaml(YAML::Node const& root);
  FricHeatOptions() = default;
  void report(std::ostream& os) const { os << "* grav = " << grav() << "\n"; }

  ADD_ARG(double, grav) = 0.;
  ADD_ARG(SedVelOptions, sedvel);
};

struct BodyHeatOptions {
  static BodyHeatOptions from_yaml(YAML::Node const& node);
  BodyHeatOptions() = default;
  void report(std::ostream& os) const {
    os << "* dTdt = " << dTdt() << "\n"
       << "* pmin = " << pmin() << "\n"
       << "* pmax = " << pmax() << "\n";
  }

  ADD_ARG(double, dTdt) = 0.0;
  ADD_ARG(double, pmin) = 0.0;
  ADD_ARG(double, pmax) = 1.0;
  ADD_ARG(kintera::ThermoOptions, thermo);
};

struct TopCoolOptions {
  static TopCoolOptions from_yaml(YAML::Node const& node);
  TopCoolOptions() = default;
  void report(std::ostream& os) const { os << "* flux = " << flux() << "\n"; }

  ADD_ARG(double, flux) = 0.0;
  ADD_ARG(CoordinateOptions, coord);
};

struct BotHeatOptions {
  static BotHeatOptions from_yaml(YAML::Node const& node);
  BotHeatOptions() = default;
  void report(std::ostream& os) const { os << "* flux = " << flux() << "\n"; }

  ADD_ARG(double, flux) = 0.0;
  ADD_ARG(CoordinateOptions, coord);
};

struct RelaxBotCompOptions {
  static RelaxBotCompOptions from_yaml(YAML::Node const& node);
  RelaxBotCompOptions() = default;
  void report(std::ostream& os) const {
    os << "* tau = " << tau() << "\n"
       << "* species = " << fmt::format("{}", species()) << "\n"
       << "* xfrac = " << fmt::format("{}", xfrac()) << "\n";
  }

  ADD_ARG(double, tau) = 0.0;
  ADD_ARG(std::vector<std::string>, species) = {};
  ADD_ARG(std::vector<double>, xfrac) = {};
};

struct RelaxBotTempOptions {
  static RelaxBotTempOptions from_yaml(YAML::Node const& node);
  RelaxBotTempOptions() = default;
  void report(std::ostream& os) const {
    os << "* tau = " << tau() << "\n"
       << "* btemp = " << btemp() << "\n";
  }

  ADD_ARG(double, tau) = 0.0;
  ADD_ARG(double, btemp) = 300.0;
};

struct RelaxBotVeloOptions {
  static RelaxBotVeloOptions from_yaml(YAML::Node const& node);
  RelaxBotVeloOptions() = default;
  void report(std::ostream& os) const {
    os << "* tau = " << tau() << "\n"
       << "* bvx = " << bvx() << "\n"
       << "* bvy = " << bvy() << "\n"
       << "* bvz = " << bvz() << "\n";
  }

  ADD_ARG(double, tau) = 0.0;
  ADD_ARG(double, bvx) = 0.0;
  ADD_ARG(double, bvy) = 0.0;
  ADD_ARG(double, bvz) = 0.0;
};

struct TopSpongeLyrOptions {
  static TopSpongeLyrOptions from_yaml(YAML::Node const& node);
  TopSpongeLyrOptions() = default;
  void report(std::ostream& os) const {
    os << "* tau = " << tau() << "\n"
       << "* width = " << width() << "\n";
  }

  ADD_ARG(double, tau) = 0.0;
  ADD_ARG(double, width) = 0.0;
  ADD_ARG(CoordinateOptions, coord);
};

struct BotSpongeLyrOptions {
  static BotSpongeLyrOptions from_yaml(YAML::Node const& node);
  BotSpongeLyrOptions() = default;
  void report(std::ostream& os) const {
    os << "* tau = " << tau() << "\n"
       << "* width = " << width() << "\n";
  }

  ADD_ARG(double, tau) = 0.0;
  ADD_ARG(double, width) = 0.0;
  ADD_ARG(CoordinateOptions, coord);
};

class ConstGravityImpl : public torch::nn::Cloneable<ConstGravityImpl> {
 public:
  //! options with which this `ConstGravity` was constructed
  ConstGravityOptions options;

  // Constructor to initialize the layers
  ConstGravityImpl() = default;
  explicit ConstGravityImpl(ConstGravityOptions const& options_)
      : options(options_) {
    reset();
  }
  void reset() override {}

  torch::Tensor forward(torch::Tensor du, torch::Tensor w, torch::Tensor temp,
                        double dt);
};
TORCH_MODULE(ConstGravity);

class Coriolis123Impl : public torch::nn::Cloneable<Coriolis123Impl> {
 public:
  //! options with which this `Coriolis123` was constructed
  CoriolisOptions options;

  // Constructor to initialize the layers
  Coriolis123Impl() = default;
  explicit Coriolis123Impl(CoriolisOptions const& options_)
      : options(options_) {
    reset();
  }
  void reset() override {}

  torch::Tensor forward(torch::Tensor du, torch::Tensor w, torch::Tensor temp,
                        double dt);
};
TORCH_MODULE(Coriolis123);

class CoriolisXYZImpl : public torch::nn::Cloneable<CoriolisXYZImpl> {
 public:
  //! options with which this `CoriolisXYZ` was constructed
  CoriolisOptions options;

  //! submodules
  Coordinate pcoord;

  // Constructor to initialize the layers
  CoriolisXYZImpl() = default;
  explicit CoriolisXYZImpl(CoriolisOptions const& options_)
      : options(options_) {
    reset();
  }
  void reset() override;

  torch::Tensor forward(torch::Tensor du, torch::Tensor w, torch::Tensor temp,
                        double dt);
};
TORCH_MODULE(CoriolisXYZ);

class DiffusionImpl : public torch::nn::Cloneable<DiffusionImpl> {
 public:
  //! options with which this `Diffusion` was constructed
  DiffusionOptions options;

  // Constructor to initialize the layers
  DiffusionImpl() = default;
  explicit DiffusionImpl(DiffusionOptions const& options_) : options(options_) {
    reset();
  }
  void reset() override {}

  torch::Tensor forward(torch::Tensor du, torch::Tensor w, torch::Tensor temp,
                        double dt);
};
TORCH_MODULE(Diffusion);

class FricHeatImpl : public torch::nn::Cloneable<FricHeatImpl> {
 public:
  //! submodules
  SedVel psedvel = nullptr;

  //! options with which this `FricHeat` was constructed
  FricHeatOptions options;

  // Constructor to initialize the layers
  FricHeatImpl() = default;
  explicit FricHeatImpl(FricHeatOptions const& options_) : options(options_) {
    reset();
  }
  void reset() override;

  torch::Tensor forward(torch::Tensor du, torch::Tensor w, torch::Tensor temp,
                        double dt);
};
TORCH_MODULE(FricHeat);

class BodyHeatImpl : public torch::nn::Cloneable<BodyHeatImpl> {
 public:
  //! submodules
  kintera::ThermoY pthermo = nullptr;

  //! options with which this `BodyHeat` was constructed
  BodyHeatOptions options;

  // Constructor to initialize the layers
  BodyHeatImpl() = default;
  explicit BodyHeatImpl(BodyHeatOptions const& options_) : options(options_) {
    reset();
  }
  void reset() override;

  torch::Tensor forward(torch::Tensor du, torch::Tensor w, torch::Tensor temp,
                        double dt);
};
TORCH_MODULE(BodyHeat);

class TopCoolImpl : public torch::nn::Cloneable<TopCoolImpl> {
 public:
  //! submodules
  Coordinate pcoord = nullptr;

  //! options with which this `TopCool` was constructed
  TopCoolOptions options;

  // Constructor to initialize the layers
  TopCoolImpl() = default;
  explicit TopCoolImpl(TopCoolOptions const& options_) : options(options_) {
    reset();
  }
  void reset() override;

  torch::Tensor forward(torch::Tensor du, torch::Tensor w, torch::Tensor temp,
                        double dt);
};
TORCH_MODULE(TopCool);

class BotHeatImpl : public torch::nn::Cloneable<BotHeatImpl> {
 public:
  //! submodules
  Coordinate pcoord = nullptr;

  //! options with which this `BotHeat` was constructed
  BotHeatOptions options;

  // Constructor to initialize the layers
  BotHeatImpl() = default;
  explicit BotHeatImpl(BotHeatOptions const& options_) : options(options_) {
    reset();
  }
  void reset() override;

  torch::Tensor forward(torch::Tensor du, torch::Tensor w, torch::Tensor temp,
                        double dt);
};
TORCH_MODULE(BotHeat);

class RelaxBotCompImpl : public torch::nn::Cloneable<RelaxBotCompImpl> {
 public:
  //! options with which this `RelaxBotComp` was constructed
  RelaxBotCompOptions options;

  // Constructor to initialize the layers
  RelaxBotCompImpl() = default;
  explicit RelaxBotCompImpl(RelaxBotCompOptions const& options_)
      : options(options_) {
    reset();
  }
  void reset() override {}

  torch::Tensor forward(torch::Tensor du, torch::Tensor w, torch::Tensor temp,
                        double dt);
};
TORCH_MODULE(RelaxBotComp);

class RelaxBotTempImpl : public torch::nn::Cloneable<RelaxBotTempImpl> {
 public:
  //! options with which this `RelaxBotTemp` was constructed
  RelaxBotTempOptions options;

  // Constructor to initialize the layers
  RelaxBotTempImpl() = default;
  explicit RelaxBotTempImpl(RelaxBotTempOptions const& options_)
      : options(options_) {
    reset();
  }
  void reset() override {}

  torch::Tensor forward(torch::Tensor du, torch::Tensor w, torch::Tensor temp,
                        double dt);
};
TORCH_MODULE(RelaxBotTemp);

class RelaxBotVeloImpl : public torch::nn::Cloneable<RelaxBotVeloImpl> {
 public:
  //! options with which this `RelaxBotVelo` was constructed
  RelaxBotVeloOptions options;

  // Constructor to initialize the layers
  RelaxBotVeloImpl() = default;
  explicit RelaxBotVeloImpl(RelaxBotVeloOptions const& options_)
      : options(options_) {
    reset();
  }
  void reset() override {}

  torch::Tensor forward(torch::Tensor du, torch::Tensor w, torch::Tensor temp,
                        double dt);
};
TORCH_MODULE(RelaxBotVelo);

class TopSpongeLyrImpl : public torch::nn::Cloneable<TopSpongeLyrImpl> {
 public:
  //! options with which this `TopSpongeLyr` was constructed
  TopSpongeLyrOptions options;

  // Constructor to initialize the layers
  TopSpongeLyrImpl() = default;
  explicit TopSpongeLyrImpl(TopSpongeLyrOptions const& options_)
      : options(options_) {
    reset();
  }
  void reset() override {}

  torch::Tensor forward(torch::Tensor du, torch::Tensor w, torch::Tensor temp,
                        double dt);
};
TORCH_MODULE(TopSpongeLyr);

class BotSpongeLyrImpl : public torch::nn::Cloneable<BotSpongeLyrImpl> {
 public:
  //! options with which this `BotSpongeLyr` was constructed
  BotSpongeLyrOptions options;

  // Constructor to initialize the layers
  BotSpongeLyrImpl() = default;
  explicit BotSpongeLyrImpl(BotSpongeLyrOptions const& options_)
      : options(options_) {
    reset();
  }
  void reset() override {}

  torch::Tensor forward(torch::Tensor du, torch::Tensor w, torch::Tensor temp,
                        double dt);
};
TORCH_MODULE(BotSpongeLyr);

}  // namespace snap

#undef ADD_ARG
