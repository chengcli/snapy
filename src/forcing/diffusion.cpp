// spdlog
#include <configure.h>
#include <spdlog/sinks/basic_file_sink.h>

// base
#include <globals.h>

// fvm
#include <fvm/index.h>

#include "forcing.hpp"

namespace snap {
torch::Tensor DiffusionImpl::forward(torch::Tensor du, torch::Tensor w,
                                     double dt) {
  // Real temp = pthermo->GetTemp(w.at(pmb->ks, j, i));
  // Real theta = potential_temp(pthermo, w.at(pmb->ks, j, i), p0);

  return du;
}
}  // namespace snap
