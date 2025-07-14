// yaml
#include <yaml-cpp/yaml.h>

// snap
#include <snap/snap.h>

#include "forcing.hpp"

namespace snap {
FricHeatOptions FricHeatOptions::from_yaml(YAML::Node const& root) {
  FricHeatOptions op;

  if (!root["forcing"]) return op;
  if (!root["forcing"]["const-gravity"]) return op;

  op.grav() = root["forcing"]["const-gravity"]["grav1"].as<double>(0.);

  if (!root["sedimentation"]) return op;

  op.sedvel() = SedVelOptions::from_yaml(root);

  return op;
}

void FricHeatImpl::reset() {
  psedvel = register_module("sedvel", SedVel(options.sedvel()));
}

torch::Tensor FricHeatImpl::forward(torch::Tensor du, torch::Tensor w,
                                    torch::Tensor temp, double dt) {
  auto dens = w[Index::IDN];
  auto pres = w[Index::IPR];

  auto vsed = psedvel->forward(dens, pres, temp);

  int ncloud = vsed.size(0);
  int nvapor = w.size(0) - 5 - ncloud;  // 5 = IDN, IPR, IVX, IVY, IVZ

  auto yfrac = w.narrow(0, Index::ICY + nvapor, ncloud);
  du[Index::IPR] += dt * w[Index::IDN] * (yfrac * vsed).sum(0) * options.grav();

  return du;
}

}  // namespace snap
