// spdlog
#include <configure.h>
#include <spdlog/sinks/basic_file_sink.h>
#include <spdlog/spdlog.h>

// base
#include <globals.h>

// fvm
#include <fvm/index.h>

#include <fvm/mesh/meshblock.hpp>
#include <fvm/registry.hpp>

#include "forcing.hpp"
#include "forcing_formatter.hpp"

namespace snap {
void Coriolis123Impl::reset() {
  LOG_INFO(logger, "{} resets with options: {}", name(), options);
}

torch::Tensor Coriolis123Impl::forward(torch::Tensor du, torch::Tensor w,
                                       double dt) {
  if (options.omega1() != 0.0 || options.omega2() != 0.0 ||
      options.omega3() != 0.0) {
    auto m1 = w[index::IDN] * w[index::IVX];
    auto m2 = w[index::IDN] * w[index::IVY];
    auto m3 = w[index::IDN] * w[index::IVZ];
    du[index::IVX] += 2. * dt * (options.omega3() * m2 - options.omega2() * m3);
    du[index::IVY] += 2. * dt * (options.omega1() * m3 - options.omega3() * m1);

    if (w.size(1) > 1) {  // 3d
      du[index::IVZ] +=
          2. * dt * (options.omega2() * m1 - options.omega1() * m2);
    }
  }

  return du;
}

void CoriolisXYZImpl::reset() {
  pcoord = register_module_op(this, "coord", options.coord());

  LOG_INFO(logger, "{} resets with options: {}", name(), options);
}

torch::Tensor CoriolisXYZImpl::forward(torch::Tensor du, torch::Tensor w,
                                       double dt) {
  if (options.omegax() != 0.0 || options.omegay() != 0.0 ||
      options.omegaz() != 0.0) {
    auto [omega1, omega2, omega3] = pcoord->vec_from_cartesian(
        {options.omegax(), options.omegay(), options.omegaz()});

    auto m1 = w[index::IDN] * w[index::IVX];
    auto m2 = w[index::IDN] * w[index::IVY];
    auto m3 = w[index::IDN] * w[index::IVZ];

    du[index::IVX] = 2. * dt * (omega3 * m2 - omega2 * m3);
    du[index::IVY] = 2. * dt * (omega1 * m3 - omega3 * m1);
    du[index::IVZ] = 2. * dt * (omega2 * m1 - omega1 * m2);
  }

  return du;
}
}  // namespace snap
