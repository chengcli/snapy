// snap
#include <snap/snap.h>

#include <snap/registry.hpp>

#include "forcing.hpp"

namespace snap {
void Coriolis123Impl::reset() {}

torch::Tensor Coriolis123Impl::forward(torch::Tensor du, torch::Tensor w,
                                       double dt) {
  if (options.omega1() != 0.0 || options.omega2() != 0.0 ||
      options.omega3() != 0.0) {
    auto m1 = w[Index::IDN] * w[Index::IVX];
    auto m2 = w[Index::IDN] * w[Index::IVY];
    auto m3 = w[Index::IDN] * w[Index::IVZ];
    du[Index::IVX] += 2. * dt * (options.omega3() * m2 - options.omega2() * m3);
    du[Index::IVY] += 2. * dt * (options.omega1() * m3 - options.omega3() * m1);

    if (w.size(1) > 1) {  // 3d
      du[Index::IVZ] +=
          2. * dt * (options.omega2() * m1 - options.omega1() * m2);
    }
  }

  return du;
}

void CoriolisXYZImpl::reset() {
  pcoord = register_module_op(this, "coord", options.coord());
}

torch::Tensor CoriolisXYZImpl::forward(torch::Tensor du, torch::Tensor w,
                                       double dt) {
  if (options.omegax() != 0.0 || options.omegay() != 0.0 ||
      options.omegaz() != 0.0) {
    auto [omega1, omega2, omega3] = pcoord->vec_from_cartesian(
        {options.omegax(), options.omegay(), options.omegaz()});

    auto m1 = w[Index::IDN] * w[Index::IVX];
    auto m2 = w[Index::IDN] * w[Index::IVY];
    auto m3 = w[Index::IDN] * w[Index::IVZ];

    du[Index::IVX] = 2. * dt * (omega3 * m2 - omega2 * m3);
    du[Index::IVY] = 2. * dt * (omega1 * m3 - omega3 * m1);
    du[Index::IVZ] = 2. * dt * (omega2 * m1 - omega1 * m2);
  }

  return du;
}
}  // namespace snap
