// spdlog
#include <configure.h>
#include <spdlog/sinks/basic_file_sink.h>

// base
#include <globals.h>

// fvm
#include <fvm/index.h>

#include "forcing.hpp"
#include "forcing_formatter.hpp"

namespace snap {
void ConstGravityImpl::reset() {
  LOG_INFO(logger, "{} resets with options: {}", name(), options);
}

torch::Tensor ConstGravityImpl::forward(torch::Tensor du, torch::Tensor w,
                                        double dt) {
  if (options.grav1() != 0.) {
    du[index::IVX] += dt * w[index::IDN] * options.grav1();
    du[index::IPR] += dt * w[index::IDN] * w[index::IVX] * options.grav1();
  }

  if (options.grav2() != 0.) {
    du[index::IVY] += dt * w[index::IDN] * options.grav2();
    du[index::IPR] += dt * w[index::IDN] * w[index::IVY] * w[index::IVY];
  }

  if (options.grav3() != 0.) {
    du[index::IVZ] += dt * w[index::IDN] * options.grav3();
    du[index::IPR] += dt * w[index::IDN] * w[index::IVZ] * w[index::IVZ];
  }

  return du;
}
}  // namespace snap
