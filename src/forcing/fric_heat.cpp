// yaml
#include <yaml-cpp/yaml.h>

// snap
#include <snap/snap.h>

#include <snap/sedimentation/sedimentation.hpp>

#include "forcing.hpp"

namespace snap {
FricHeatOptions FricHeatOptionsImpl::from_yaml(YAML::Node const& forcing) {
  if (!forcing["fric-heat"]) return nullptr;

  auto node = forcing["fric-heat"];
  auto op = FricHeatOptionsImpl::create();

  return op;
}

void FricHeatImpl::reset() {
  psedvel = SedVelImpl::create(options->sedvel(), this);
}

torch::Tensor FricHeatImpl::forward(torch::Tensor du, torch::Tensor w,
                                    torch::Tensor temp, double dt) {
  auto dens = w[IDN];
  auto pres = w[IPR];

  auto vsed = psedvel->forward(dens, pres, temp);

  int ncloud = vsed.size(0);
  int nvapor = w.size(0) - 5 - ncloud;  // 5 = IDN, IPR, IVX, IVY, IVZ

  auto yfrac = w.narrow(0, ICY + nvapor, ncloud);
  auto grav = -options->sedvel()->grav()->grav1();
  du[IPR] += dt * w[IDN] * (yfrac * vsed).sum(0) * grav;

  return du;
}

}  // namespace snap
